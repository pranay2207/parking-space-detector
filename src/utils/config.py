#!/usr/bin/env python3
import os
from pathlib import Path
import yaml

DEFAULT_CONFIG = {
    "detection": {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
    },
    "paths": {
        "models_dir": "models",
        "data_dir": "data",
        "spaces_file": "data/spaces.json",
        "occupancy_dir": "data/occupancy"
    }
}

class Config:
    def __init__(self, config_path=None):
        """
        Initialize configuration manager.
        
        Args:
            config_path (str, optional): Path to config file
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    # Update config with loaded data
                    self._deep_update(self.config, config_data)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Error loading config file: {e}")
    
    def _deep_update(self, target, source):
        """Recursively update nested dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def get(self, path, default=None):
        """
        Get a configuration value using dot notation.
        
        Args:
            path (str): Path to configuration value (e.g., 'detection.confidence_threshold')
            default: Default value if path not found
            
        Returns:
            Value at the specified path or default
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, path, value):
        """
        Set a configuration value using dot notation.
        
        Args:
            path (str): Path to configuration value (e.g., 'detection.confidence_threshold')
            value: Value to set
        """
        keys = path.split('.')
        target = self.config
        
        # Navigate to the final container
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
            
        # Set the value
        target[keys[-1]] = value
    
    def save(self, config_path):
        """Save configuration to file."""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving config file: {e}")
            return False 