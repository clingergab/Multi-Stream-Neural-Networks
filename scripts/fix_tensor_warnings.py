#!/usr/bin/env python
"""
Enhanced script to fix PyTorch tensor warnings in the Multi-Stream Neural Networks codebase.
Fixes multiple patterns of incorrect tensor construction:
1. torch.tensor(tensor) -> tensor.detach().clone()
2. torch.tensor(np_array) -> torch.from_numpy(np_array)
3. Fixes indentation issues in complex conditional statements
"""

import re
import os

def fix_tensor_warnings(file_path):
    """Fix PyTorch tensor warnings by replacing improper tensor creation."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Pattern 1: Simple tensor creation with an existing tensor
    pattern1 = r'(\w+_tensor)\s*=\s*torch\.tensor\((\w+_data),\s*dtype=torch\.(float32|long)\)'
    replacement1 = r'if isinstance(\2, np.ndarray):\n                \1 = torch.from_numpy(\2).float() if "\3" == "float32" else torch.from_numpy(\2).long()\n            elif isinstance(\2, torch.Tensor):\n                \1 = \2.detach().clone()\n            else:\n                \1 = \2'
    
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: Fix the broken if-else blocks in predict and predict_proba methods
    # This is a more aggressive fix using string replacement instead of regex
    broken_pattern = """            # Convert to tensors if needed
            if isinstance(color_data, np.ndarray):
                if isinstance(color_data, torch.Tensor):
                color_tensor = color_data.detach().clone()
            else:
                color_tensor = torch.tensor(color_data, dtype=torch.float32)
            else:
                color_tensor = color_data"""
                
    fixed_replacement = """            # Convert to tensors if needed
            if isinstance(color_data, np.ndarray):
                color_tensor = torch.from_numpy(color_data).float()
            elif isinstance(color_data, torch.Tensor):
                color_tensor = color_data.detach().clone()
            else:
                color_tensor = color_data"""
                
    content = content.replace(broken_pattern, fixed_replacement)
    
    # Similar pattern for brightness data
    broken_pattern2 = """            if isinstance(brightness_data, np.ndarray):
                if isinstance(brightness_data, torch.Tensor):
                brightness_tensor = brightness_data.detach().clone()
            else:
                brightness_tensor = torch.tensor(brightness_data, dtype=torch.float32)
            else:
                brightness_tensor = brightness_data"""
                
    fixed_replacement2 = """            if isinstance(brightness_data, np.ndarray):
                brightness_tensor = torch.from_numpy(brightness_data).float()
            elif isinstance(brightness_data, torch.Tensor):
                brightness_tensor = brightness_data.detach().clone()
            else:
                brightness_tensor = brightness_data"""
                
    content = content.replace(broken_pattern2, fixed_replacement2)
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed tensor warnings in {file_path}")

def main():
    """Fix tensor warnings in key model files."""
    # Base model file
    base_path = '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src/models/basic_multi_channel/base_multi_channel_network.py'
    fix_tensor_warnings(base_path)
    
    # ResNet model file
    resnet_path = '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src/models/basic_multi_channel/multi_channel_resnet_network.py'
    fix_tensor_warnings(resnet_path)
    
    print("Fixed all tensor warnings in model files.")

if __name__ == "__main__":
    main()
