#!/usr/bin/env python3
"""
Script to update notebook API calls from old methods to new simplified API.
This replaces all instances of:
- model.forward_combined() -> model()
- forward_combined -> forward (in documentation)
"""

import json
import re

def update_notebook_api():
    """Update the notebook to use the new simplified API."""
    notebook_path = '/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/Multi_Stream_CIFAR100_Training.ipynb'
    
    print("📝 Reading notebook...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Track changes
    changes_made = 0
    
    print("🔄 Processing cells...")
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Process code cells
            for i, line in enumerate(cell['source']):
                original_line = line
                
                # Replace .forward_combined( with (
                line = re.sub(r'\.forward_combined\(', '(', line)
                
                # Update the line if it changed
                if line != original_line:
                    cell['source'][i] = line
                    changes_made += 1
                    print(f"   ✅ Code: {original_line.strip()} -> {line.strip()}")
        
        elif cell['cell_type'] == 'markdown':
            # Process markdown cells for documentation updates
            for i, line in enumerate(cell['source']):
                original_line = line
                
                # Replace documentation references
                line = re.sub(r'`model\.forward_combined\(\)`', '`model()`', line)
                line = re.sub(r'forward_combined\(\)', 'forward()', line)
                line = re.sub(r'Standard classification output \(single logits tensor\)', 'Primary method for training, inference, and evaluation', line)
                
                # Update the line if it changed
                if line != original_line:
                    cell['source'][i] = line
                    changes_made += 1
                    print(f"   ✅ Docs: {original_line.strip()} -> {line.strip()}")
    
    if changes_made > 0:
        print(f"\n💾 Saving notebook with {changes_made} changes...")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("✅ Notebook updated successfully!")
    else:
        print("ℹ️  No changes needed - notebook is already up to date!")

if __name__ == "__main__":
    update_notebook_api()
