# Styled QR Code Generator 🎨

Generate beautiful, scannable QR codes styled after any image while maintaining full functionality.

## Features

- 🎨 **Image-Based Styling**: Extract colors and shapes from any image
- 🔍 **Scannable**: Maintains QR code functionality with high error correction
- 🎯 **Multiple Styles**: Square, rounded, circular, or gapped modules
- 🖼️ **Center Embedding**: Optionally embed your image in the QR center
- ⚙️ **Highly Configurable**: YAML/JSON config support
- 🧪 **Well Tested**: Comprehensive test suite included
- 📝 **Well Documented**: Clear code with extensive comments

## Installation

```bash
pip install qrcode[pil] pillow numpy opencv-python scikit-image

**QUICK START** 

from styled_qr_generator import StyledQRGenerator

generator = StyledQRGenerator()
qr_image = generator.generate(
    data="https://your-url.com",
    image_path="banana.jpg",
    output_path="banana_qr.png",
    embed_image=True
)

**Command Line Usage**
# Basic usage
python cli.py -d "https://example.com" -i banana.jpg -o banana_qr.png

# With embedded image
python cli.py -d "Hello" -i logo.png -o qr.png --embed

# Custom style
python cli.py -d "Data" -i pic.jpg -o out.png --style circular --intensity 0.8

How It Works

    Color Extraction: Uses K-means clustering to find dominant colors
    Shape Detection: Applies edge detection and contour finding
    QR Generation: Creates QR with high error correction (30%)
    Style Application: Applies colors and shapes while preserving finder patterns
    Contrast Enhancement: Ensures scannability

Configuration

Create a config.yaml:

qr_config:
  error_correction: H
  box_size: 10
  border: 4

style_config:
  style_intensity: 0.6
  module_style: rounded
  preserve_finder_patterns: true

Best Practices

    Use High Error Correction: Set to 'H' for 30% error correction
    Test Thoroughly: Scan with multiple apps and devices
    Maintain Contrast: Ensure dark/light modules have sufficient contrast
    Preserve Patterns: Keep finder patterns (corners) clear
    Size Appropriately: Larger QR codes scan more reliably

Architecture

    styled_qr_generator.py: Main generator logic
    config.py: Configuration management
    cli.py: Command-line interface
    test_qr_generator.py: Test suite

License

MIT License


This implementation provides:

1. **Comprehensive solution** with modular architecture
2. **High-quality code** with type hints and documentation
3. **Flexible configuration** via code or config files
4. **CLI support** for easy usage
5. **Testing suite** for reliability
6. **Best practices** for QR code scannability

The key innovation is balancing artistic styling with QR code functionality by:
- Using maximum error correction
- Preserving critical patterns
- Maintaining sufficient contrast
- Applying strategic masking