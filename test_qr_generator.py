"""
test_qr_generator.py - Unit tests for QR generator
"""

import unittest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

from qr import (
    StyledQRGenerator,
    ImageAnalyzer,
    QRConfig,
    StyleConfig
)


class TestImageAnalyzer(unittest.TestCase):
    """Test image analysis functionality"""
    
    def setUp(self):
        """Create test images"""
        self.test_dir = tempfile.mkdtemp()
        self.analyzer = ImageAnalyzer()
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        self.test_image_path = os.path.join(self.test_dir, 'test.png')
        img.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        os.rmdir(self.test_dir)
    
    def test_extract_dominant_colors(self):
        """Test color extraction"""
        colors = self.analyzer.extract_dominant_colors(self.test_image_path, n_colors=3)
        
        self.assertIsInstance(colors, list)
        self.assertGreater(len(colors), 0)
        self.assertEqual(len(colors[0]), 3)  # RGB tuple
    
    def test_get_color_palette(self):
        """Test palette generation"""
        palette = self.analyzer.get_color_palette(self.test_image_path)
        
        self.assertIn('dark', palette)
        self.assertIn('light', palette)
        self.assertIsInstance(palette['dark'], tuple)
        self.assertEqual(len(palette['dark']), 3)
    
    def test_extract_shape_mask(self):
        """Test shape mask extraction"""
        mask = self.analyzer.extract_shape_mask(self.test_image_path, (50, 50))
        
        self.assertIsInstance(mask, Image.Image)
        self.assertEqual(mask.size, (50, 50))


class TestStyledQRGenerator(unittest.TestCase):
    """Test QR code generation"""
    
    def setUp(self):
        """Initialize test environment"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test image
        img = Image.new('RGB', (200, 200), color='blue')
        self.test_image_path = os.path.join(self.test_dir, 'test.png')
        img.save(self.test_image_path)
        
        self.generator = StyledQRGenerator()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        
        # Clean up generated QR codes
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)
    
    def test_generate_basic_qr(self):
        """Test basic QR code generation"""
        output_path = os.path.join(self.test_dir, 'output.png')
        
        result = self.generator.generate(
            data="Test Data",
            image_path=self.test_image_path,
            output_path=output_path,
            embed_image=False
        )
        
        self.assertIsInstance(result, Image.Image)
        self.assertTrue(os.path.exists(output_path))
    
    def test_generate_with_embed(self):
        """Test QR generation with embedded image"""
        output_path = os.path.join(self.test_dir, 'output_embed.png')
        
        result = self.generator.generate(
            data="https://example.com",
            image_path=self.test_image_path,
            output_path=output_path,
            embed_image=True
        )
        
        self.assertTrue(os.path.exists(output_path))
        self.assertIsInstance(result, Image.Image)
    
    def test_different_styles(self):
        """Test different module styles"""
        styles = ['square', 'rounded', 'circular', 'gapped']
        
        for style in styles:
            with self.subTest(style=style):
                style_config = StyleConfig(module_style=style)
                generator = StyledQRGenerator(style_config=style_config)
                
                output_path = os.path.join(self.test_dir, f'output_{style}.png')
                
                result = generator.generate(
                    data="Test",
                    image_path=self.test_image_path,
                    output_path=output_path
                )
                
                self.assertTrue(os.path.exists(output_path))


if __name__ == '__main__':
    unittest.main()
