#!/usr/bin/env python
"""
Manual patch to fix indentation errors in tensor warning fixes.
"""

def fix_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for problematic pattern: else followed by identical if check
        if line.strip().startswith('else:') and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line.strip().startswith('if isinstance(') and 'detach().clone()' in lines[i + 2]:
                # Skip the redundant if check
                i += 2
            else:
                new_lines.append(line)
                i += 1
        else:
            new_lines.append(line)
            i += 1
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Fixed indentation errors in {file_path}")

if __name__ == "__main__":
    base_path = '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src/models/basic_multi_channel/base_multi_channel_network.py'
    resnet_path = '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/src/models/basic_multi_channel/multi_channel_resnet_network.py'
    
    fix_file(base_path)
    fix_file(resnet_path)
    
    print("Manual patch completed.")
