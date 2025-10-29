import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import (
    RoundedModuleDrawer, 
    CircleModuleDrawer,
    GappedSquareModuleDrawer
)
from qrcode.image.styles.colormasks import SolidFillColorMask
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import warnings

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QRConfig:
    """Configuration for QR code generation"""
    version: Optional[int] = None  # None for automatic
    error_correction: int = qrcode.constants.ERROR_CORRECT_H  # Highest error correction
    box_size: int = 10  # Size of each QR code box in pixels
    border: int = 4  # Border size in boxes
    

@dataclass
class StyleConfig:
    """Configuration for visual styling"""
    style_intensity: float = 0.5  # How much to apply image style (0-1)
    contrast_threshold: float = 0.4  # Minimum contrast for scannability
    preserve_finder_patterns: bool = True  # Keep finder patterns clear
    module_style: str = "rounded"  # Options: square, rounded, circular, gapped
    use_image_overlay: bool = False  # Use image as overlay (experimental)
    

class ImageAnalyzer:
    """Analyzes input images to extract visual characteristics"""
    
    @staticmethod
    def extract_dominant_colors(image_path: str, n_colors: int = 5) -> list:
        """
        Extract dominant colors from image using K-means clustering
        
        Args:
            image_path: Path to input image
            n_colors: Number of dominant colors to extract
            
        Returns:
            List of RGB tuples representing dominant colors
        """
        logger.info(f"Extracting dominant colors from {image_path}")
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape image to be a list of pixels
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert to integers and sort by frequency
        centers = np.uint8(centers)
        unique, counts = np.unique(labels, return_counts=True)
        
        # Sort colors by frequency
        sorted_indices = np.argsort(-counts)
        dominant_colors = [tuple(map(int, centers[i])) for i in sorted_indices]
        
        logger.info(f"Dominant colors: {dominant_colors}")
        return dominant_colors
    
    @staticmethod
    def get_color_palette(image_path: str) -> Dict[str, Tuple[int, int, int]]:
        """
        Get a color palette suitable for QR codes (dark and light colors)
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with 'dark' and 'light' color keys
        """
        colors = ImageAnalyzer.extract_dominant_colors(image_path, n_colors=6)
        
        # Calculate luminance for each color
        def luminance(rgb):
            return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        
        colors_with_lum = [(c, luminance(c)) for c in colors]
        colors_with_lum.sort(key=lambda x: x[1])
        
        # Select darkest and lightest colors
        dark_color = colors_with_lum[0][0]
        light_color = colors_with_lum[-1][0]
        
        # Ensure sufficient contrast (minimum difference of 128 in luminance)
        contrast = abs(colors_with_lum[-1][1] - colors_with_lum[0][1])
        if contrast < 128:
            logger.warning(f"Low contrast detected ({contrast:.1f}), adjusting colors")
            # Darken the dark color and lighten the light color
            dark_color = tuple(min(int(c * 0.3), 50) for c in dark_color)
            light_color = tuple(max(int(c * 1.3), 200) for c in light_color)
        
        # Try to find a mid-tone accent color
        accent_color = colors[len(colors) // 2] if len(colors) > 2 else dark_color
        
        return {
            'dark': dark_color,
            'light': light_color,
            'accent': accent_color
        }
    
    @staticmethod
    def extract_shape_mask(image_path: str, target_size: Tuple[int, int]) -> Image.Image:
        """
        Extract the main subject's shape as a mask
        
        Args:
            image_path: Path to input image
            target_size: Size to resize mask to
            
        Returns:
            PIL Image mask
        """
        logger.info(f"Extracting shape mask from {image_path}")
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create filled shape
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours and fill
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray)
        if contours:
            # Fill the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        else:
            # If no contours, use the whole image
            mask = np.ones_like(gray) * 255
        
        # Convert to PIL and resize
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Apply blur for smooth edges
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
        
        return mask_img


class SafeColorMask(SolidFillColorMask):
    """
    Safe color mask implementation that handles edge cases
    """
    
    def __init__(self, back_color: Tuple[int, int, int], 
                 front_color: Tuple[int, int, int]):
        """
        Initialize color mask with validation
        
        Args:
            back_color: Background (light) color
            front_color: Foreground (dark) color
        """
        # Ensure colors are tuples of ints
        self.back_color = tuple(int(c) for c in back_color)
        self.front_color = tuple(int(c) for c in front_color)
        
        # Call parent with validated colors
        super().__init__(back_color=self.back_color, front_color=self.front_color)


class StyledQRGenerator:
    """Main QR code generator with image-based styling"""
    
    def __init__(self, qr_config: QRConfig = None, style_config: StyleConfig = None):
        """
        Initialize QR generator
        
        Args:
            qr_config: QR code configuration
            style_config: Styling configuration
        """
        self.qr_config = qr_config or QRConfig()
        self.style_config = style_config or StyleConfig()
        self.image_analyzer = ImageAnalyzer()
    
    def _get_module_drawer(self, style: str):
        """Get appropriate module drawer based on style"""
        drawers = {
            'rounded': RoundedModuleDrawer(),
            'circular': CircleModuleDrawer(),
            'gapped': GappedSquareModuleDrawer(),
            'square': None  # Default
        }
        return drawers.get(style, None)
    
    def _create_base_qr(self, data: str) -> qrcode.QRCode:
        """
        Create base QR code object
        
        Args:
            data: Data to encode in QR code
            
        Returns:
            QRCode object
        """
        qr = qrcode.QRCode(
            version=self.qr_config.version,
            error_correction=self.qr_config.error_correction,
            box_size=self.qr_config.box_size,
            border=self.qr_config.border,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        return qr
    
    def _extract_image_from_qr_object(self, qr_obj) -> Image.Image:
        """
        Safely extract PIL Image from QR code object
        
        Args:
            qr_obj: QR code image object (could be StyledPilImage or PIL Image)
            
        Returns:
            PIL Image
        """
        # Check if it's already a PIL Image
        if isinstance(qr_obj, Image.Image):
            return qr_obj
        
        # Try to get _img attribute (StyledPilImage stores PIL Image here)
        if hasattr(qr_obj, '_img'):
            return qr_obj._img
        
        # Try get_image method
        if hasattr(qr_obj, 'get_image'):
            return qr_obj.get_image()
        
        # Try converting directly
        if hasattr(qr_obj, 'convert'):
            return qr_obj.convert('RGB')
        
        raise TypeError(f"Could not extract PIL Image from QR object of type {type(qr_obj)}")
    
    def _apply_shape_mask(self, qr_img: Image.Image, shape_mask: Image.Image) -> Image.Image:
        """
        Apply shape mask to QR code while preserving critical patterns
        
        Args:
            qr_img: QR code image
            shape_mask: Shape mask to apply
            
        Returns:
            Masked QR code image
        """
        # Ensure qr_img is valid
        if qr_img is None or not hasattr(qr_img, 'size'):
            raise ValueError("Invalid QR code image")
        
        # Resize mask to QR code size
        shape_mask = shape_mask.resize(qr_img.size, Image.Resampling.LANCZOS)
        
        # Convert to RGBA if needed
        if qr_img.mode != 'RGBA':
            qr_img = qr_img.convert('RGBA')
        
        # Create a protective mask for finder patterns (corners)
        protection_mask = Image.new('L', qr_img.size, 255)
        draw = ImageDraw.Draw(protection_mask)
        
        if self.style_config.preserve_finder_patterns:
            # Calculate finder pattern positions (approximate)
            # Finder patterns are 7 modules wide with 4 module border
            module_size = self.qr_config.box_size
            border_pixels = self.qr_config.border * module_size
            pattern_size = 7 * module_size  # 7x7 modules
            padding = int(pattern_size * 1.5)  # Extra padding for safety
            
            # Top-left
            draw.rectangle([0, 0, border_pixels + padding, border_pixels + padding], fill=0)
            # Top-right
            draw.rectangle([
                qr_img.size[0] - border_pixels - padding, 0,
                qr_img.size[0], border_pixels + padding
            ], fill=0)
            # Bottom-left
            draw.rectangle([
                0, qr_img.size[1] - border_pixels - padding,
                border_pixels + padding, qr_img.size[1]
            ], fill=0)
        
        # Combine protection mask with shape mask
        shape_mask_array = np.array(shape_mask.convert('L'))
        protection_mask_array = np.array(protection_mask)
        
        # Where protection mask is 0 (protected areas), keep full opacity
        # Where protection mask is 255 (non-protected), use shape mask
        combined_mask = np.where(
            protection_mask_array == 0,
            255,
            shape_mask_array
        ).astype(np.uint8)
        
        combined_mask_img = Image.fromarray(combined_mask)
        
        # Apply combined mask
        qr_img.putalpha(combined_mask_img)
        
        return qr_img
    
    def _enhance_contrast(self, img: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Enhance image contrast for better scannability
        
        Args:
            img: Input image
            factor: Contrast enhancement factor
            
        Returns:
            Enhanced image
        """
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def _create_simple_qr(self, 
                         qr: qrcode.QRCode,
                         fill_color: Tuple[int, int, int],
                         back_color: Tuple[int, int, int]) -> Image.Image:
        """
        Create a simple QR code without fancy styling
        
        Args:
            qr: QRCode object
            fill_color: Foreground color
            back_color: Background color
            
        Returns:
            PIL Image
        """
        logger.info("Creating simple styled QR code")
        
        # Create basic image
        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        
        # Convert to PIL Image if needed
        if not isinstance(img, Image.Image):
            if hasattr(img, '_img'):
                img = img._img
            else:
                # Fallback: create from matrix
                matrix = qr.get_matrix()
                size = (len(matrix[0]) * self.qr_config.box_size + 2 * self.qr_config.border * self.qr_config.box_size,
                       len(matrix) * self.qr_config.box_size + 2 * self.qr_config.border * self.qr_config.box_size)
                img = Image.new('RGB', size, back_color)
                draw = ImageDraw.Draw(img)
                
                offset = self.qr_config.border * self.qr_config.box_size
                for r, row in enumerate(matrix):
                    for c, val in enumerate(row):
                        if val:
                            box = [
                                c * self.qr_config.box_size + offset,
                                r * self.qr_config.box_size + offset,
                                (c + 1) * self.qr_config.box_size + offset,
                                (r + 1) * self.qr_config.box_size + offset
                            ]
                            draw.rectangle(box, fill=fill_color)
        
        return img
    
    def generate(self, 
                data: str,
                image_path: str,
                output_path: str,
                embed_image: bool = False) -> Image.Image:
        """
        Generate styled QR code based on input image
        
        Args:
            data: Data to encode in QR code (URL, text, etc.)
            image_path: Path to style reference image
            output_path: Path to save output QR code
            embed_image: Whether to embed the original image in center
            
        Returns:
            Generated QR code as PIL Image
        """
        logger.info(f"Generating QR code for: {data}")
        logger.info(f"Using style from: {image_path}")
        
        try:
            # Analyze input image
            color_palette = self.image_analyzer.get_color_palette(image_path)
            logger.info(f"Color palette: {color_palette}")
            
            # Create base QR code
            qr = self._create_base_qr(data)
            
            # Get module drawer
            module_drawer = self._get_module_drawer(self.style_config.module_style)
            
            # Generate QR code image
            if module_drawer and self.style_config.module_style != 'square':
                try:
                    logger.info(f"Creating QR with {self.style_config.module_style} modules")
                    
                    # Create color mask
                    color_mask = SafeColorMask(
                        back_color=color_palette['light'],
                        front_color=color_palette['dark']
                    )
                    
                    # Generate with style
                    qr_obj = qr.make_image(
                        image_factory=StyledPilImage,
                        module_drawer=module_drawer,
                        color_mask=color_mask
                    )
                    
                    # Extract PIL Image
                    qr_img = self._extract_image_from_qr_object(qr_obj)
                    
                except Exception as e:
                    logger.warning(f"Styled QR generation failed: {e}. Falling back to simple style.")
                    qr_img = self._create_simple_qr(qr, color_palette['dark'], color_palette['light'])
            else:
                # Create simple QR
                qr_img = self._create_simple_qr(qr, color_palette['dark'], color_palette['light'])
            
            # Verify we have a valid image
            if qr_img is None or not hasattr(qr_img, 'size'):
                raise ValueError("Failed to generate QR code image")
            
            logger.info(f"QR code base image created: {qr_img.size}, mode: {qr_img.mode}")
            
            # Extract and apply shape mask
            if self.style_config.style_intensity > 0:
                try:
                    shape_mask = self.image_analyzer.extract_shape_mask(
                        image_path, 
                        qr_img.size
                    )
                    qr_img = self._apply_shape_mask(qr_img, shape_mask)
                    logger.info("Shape mask applied successfully")
                except Exception as e:
                    logger.warning(f"Could not apply shape mask: {e}")
            
            # Create final image with background
            final_img = Image.new('RGB', qr_img.size, color_palette['light'])
            
            # Paste QR code (with alpha if available)
            if qr_img.mode == 'RGBA':
                final_img.paste(qr_img, (0, 0), qr_img)
            else:
                final_img.paste(qr_img, (0, 0))
            
            # Embed center image if requested
            if embed_image:
                try:
                    final_img = self._embed_center_image(final_img, image_path)
                    logger.info("Center image embedded")
                except Exception as e:
                    logger.warning(f"Could not embed center image: {e}")
            
            # Enhance contrast for better scanning
            contrast_factor = 1.0 + self.style_config.contrast_threshold
            final_img = self._enhance_contrast(final_img, contrast_factor)
            
            # Save
            final_img.save(output_path, quality=95)
            logger.info(f"QR code saved to: {output_path}")
            
            return final_img
            
        except Exception as e:
            logger.error(f"Error during QR generation: {e}", exc_info=True)
            raise
    
    def _embed_center_image(self, qr_img: Image.Image, image_path: str) -> Image.Image:
        """
        Embed the original image in the center of QR code
        
        Args:
            qr_img: QR code image
            image_path: Path to image to embed
            
        Returns:
            QR code with embedded image
        """
        # Open and prepare center image
        center_img = Image.open(image_path)
        center_img = center_img.convert('RGBA')
        
        # Calculate size (should be about 20-25% of QR code for H error correction)
        qr_width, qr_height = qr_img.size
        center_size = int(min(qr_width, qr_height) * 0.22)
        
        # Resize center image maintaining aspect ratio
        center_img.thumbnail((center_size, center_size), Image.Resampling.LANCZOS)
        
        # Calculate position (center)
        center_x = (qr_width - center_img.width) // 2
        center_y = (qr_height - center_img.height) // 2
        
        # Add white border around center image
        border_size = 8
        bordered_size = (
            center_img.width + border_size * 2,
            center_img.height + border_size * 2
        )
        bordered_img = Image.new('RGB', bordered_size, (255, 255, 255))
        
        # Paste center image on white background
        if center_img.mode == 'RGBA':
            bordered_img.paste(center_img, (border_size, border_size), center_img)
        else:
            bordered_img.paste(center_img, (border_size, border_size))
        
        # Paste on QR code
        qr_img.paste(
            bordered_img,
            (center_x - border_size, center_y - border_size)
        )
        
        return qr_img


def main():
    """Example usage of the styled QR generator"""
    
    print("=" * 60)
    print("Custom QR Code Generator - Fixed Version")
    print("=" * 60)
    
    # Example usage
    print("\nUsage Example:")
    print("-" * 60)
    print("""
from styled_qr_generator import StyledQRGenerator, QRConfig, StyleConfig

# Create generator with configuration
qr_config = QRConfig(
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4
)

style_config = StyleConfig(
    style_intensity=0.5,
    module_style="rounded",  # rounded, circular, gapped, square
    preserve_finder_patterns=True
)

generator = StyledQRGenerator(qr_config, style_config)

# Generate QR code
qr_image = generator.generate(
    data="https://your-url.com",
    image_path="banana.jpg",
    output_path="banana_qr.png",
    embed_image=True
)

print("QR code generated successfully!")
    """)
    
    print("\nTroubleshooting Tips:")
    print("-" * 60)
    print("✓ Ensure all dependencies are installed")
    print("✓ Use high-contrast source images for best results")
    print("✓ Test generated QR codes with multiple scanner apps")
    print("✓ If styling fails, the generator falls back to simple QR")
    print("✓ Reduce style_intensity if QR code doesn't scan well")
    

if __name__ == "__main__":
    main()
