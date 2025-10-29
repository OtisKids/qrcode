"""
cli.py - Fixed and simplified CLI
"""

import argparse
import sys
from pathlib import Path
import qrcode

# Import after ensuring path is correct
try:
    from qr import (
        StyledQRGenerator, 
        QRConfig, 
        StyleConfig
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure styled_qr_generator.py is in the same directory")
    sys.exit(1)


def parse_error_correction(ec_str: str) -> int:
    """Convert error correction string to constant"""
    ec_map = {
        'L': qrcode.constants.ERROR_CORRECT_L,
        'M': qrcode.constants.ERROR_CORRECT_M,
        'Q': qrcode.constants.ERROR_CORRECT_Q,
        'H': qrcode.constants.ERROR_CORRECT_H,
    }
    return ec_map.get(ec_str.upper(), qrcode.constants.ERROR_CORRECT_H)


def main():
    parser = argparse.ArgumentParser(
        description='Generate styled QR codes based on input images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py -d "https://example.com" -i banana.jpg -o banana_qr.png
  
  # With embedded center image
  python cli.py -d "Hello World" -i logo.png -o styled_qr.png --embed
  
  # Custom styling
  python cli.py -d "Data" -i pic.jpg -o out.png --style rounded --intensity 0.5
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-d', '--data',
        required=True,
        help='Data to encode in QR code (URL, text, etc.)'
    )
    parser.add_argument(
        '-i', '--image',
        required=True,
        help='Input image for styling'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output QR code image path'
    )
    
    # Optional arguments
    parser.add_argument(
        '--embed',
        action='store_true',
        help='Embed input image in QR code center'
    )
    parser.add_argument(
        '--style',
        choices=['square', 'rounded', 'circular', 'gapped'],
        default='rounded',
        help='QR code module style (default: rounded)'
    )
    parser.add_argument(
        '--intensity',
        type=float,
        default=0.5,
        help='Style intensity 0.0 to 1.0 (default: 0.5)'
    )
    parser.add_argument(
        '--error-correction',
        choices=['L', 'M', 'Q', 'H'],
        default='H',
        help='Error correction level (default: H - highest)'
    )
    parser.add_argument(
        '--box-size',
        type=int,
        default=10,
        help='Size of each QR module in pixels (default: 10)'
    )
    parser.add_argument(
        '--no-preserve-patterns',
        action='store_true',
        help='Do not preserve finder patterns (may reduce scannability)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    # Validate intensity
    if not 0 <= args.intensity <= 1:
        print(f"Error: Intensity must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Create configurations
    qr_config = QRConfig(
        error_correction=parse_error_correction(args.error_correction),
        box_size=args.box_size,
        border=4
    )
    
    style_config = StyleConfig(
        style_intensity=args.intensity,
        module_style=args.style,
        preserve_finder_patterns=not args.no_preserve_patterns,
        use_image_overlay=False  # Keep this false for stability
    )
    
    # Create generator
    generator = StyledQRGenerator(qr_config, style_config)
    
    # Generate QR code
    try:
        print(f"\n{'='*60}")
        print(f"Generating Styled QR Code")
        print(f"{'='*60}")
        print(f"  Data: {args.data}")
        print(f"  Style image: {args.image}")
        print(f"  Output: {args.output}")
        print(f"  Module style: {args.style}")
        print(f"  Style intensity: {args.intensity}")
        print(f"  Error correction: {args.error_correction} (30% for H)")
        print(f"  Embed image: {args.embed}")
        print(f"{'='*60}\n")
        
        qr_image = generator.generate(
            data=args.data,
            image_path=args.image,
            output_path=args.output,
            embed_image=args.embed
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Success!")
        print(f"{'='*60}")
        print(f"QR code saved to: {args.output}")
        print(f"Image size: {qr_image.size}")
        print(f"\nNext steps:")
        print(f"  1. Open the QR code image to verify it looks good")
        print(f"  2. Test with multiple QR scanner apps")
        print(f"  3. If it doesn't scan, try:")
        print(f"     - Lower intensity (--intensity 0.3)")
        print(f"     - Use 'square' style (--style square)")
        print(f"     - Don't embed image (remove --embed)")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Error generating QR code")
        print(f"{'='*60}")
        print(f"{e}")
        print(f"\nTroubleshooting:")
        print(f"  - Check that the image file is valid")
        print(f"  - Try with a simpler image")
        print(f"  - Use lower intensity value")
        print(f"  - Try --style square for most reliable generation")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
