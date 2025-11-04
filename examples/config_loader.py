"""
Configuration loader and saver for SNN network settings

Author: Jiwoon Lee (@metr0jw)
Organization: Kwangwoon University, Seoul, South Korea
Contact: jwlee@linux.com
"""
import json
import yaml

def load_network_config(filename: str) -> dict:
    """Load network configuration from file"""
    if filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    elif filename.endswith('.yaml'):
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format")

def save_network_config(config: dict, filename: str):
    """Save network configuration to file"""
    if filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    elif filename.endswith('.yaml'):
        with open(filename, 'w') as f:
            yaml.dump(config, f)
