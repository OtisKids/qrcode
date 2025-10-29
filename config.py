"""
config.py - Configuration management for QR generator
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages configuration loading and saving"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        path = Path(config_path)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to file"""
        path = Path(config_path)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'qr_config': {
                'version': None,
                'error_correction': 'H',
                'box_size': 10,
                'border': 4
            },
            'style_config': {
                'style_intensity': 0.6,
                'contrast_threshold': 0.4,
                'preserve_finder_patterns': True,
                'module_style': 'rounded',
                'use_gradient': False
            }
        }


# Example config.yaml
"""
qr_config:
  version: null  # Auto-detect
  error_correction: H  # L, M, Q, H
  box_size: 10
  border: 4

style_config:
  style_intensity: 0.6  # 0.0 to 1.0
  contrast_threshold: 0.4
  preserve_finder_patterns: true
  module_style: rounded  # square, rounded, circular, gapped
  use_gradient: false
"""
