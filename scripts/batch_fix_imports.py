#!/usr/bin/env python3
"""
Batch fix script for common 'from src.' import patterns.
"""

import os
import re
import sys
from pathlib import Path

def fix_file_imports(file_path):
    """Fix imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Common import patterns to fix
        patterns = [
            # Basic imports
            (r'from src\.data_utils\.', 'from data_utils.'),
            (r'from src\.models\.', 'from models.'),
            (r'from src\.models2\.', 'from models2.'),
            (r'from src\.utils\.', 'from utils.'),
            (r'from src\.transforms\.', 'from transforms.'),
            
            # Specific common imports
            (r'from src\.data_utils import', 'from data_utils import'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Check if we need to add sys.path setup for scripts
        if content != original_content and ('/scripts/' in file_path or '/tests/' in file_path):
            # Check if file already has sys.path setup
            if 'sys.path' not in content and 'import sys' in content:
                # Add sys.path setup after existing imports
                lines = content.split('\n')
                import_end = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_end = i
                
                # Insert sys.path setup after imports
                if import_end > 0:
                    path_setup = [
                        '',
                        '# Add src to path for imports',
                        'from pathlib import Path',
                        'project_root = Path(__file__).parent.parent',
                        'sys.path.insert(0, str(project_root))',
                        'sys.path.insert(0, str(project_root / "src"))',
                        ''
                    ]
                    lines = lines[:import_end+1] + path_setup + lines[import_end+1:]
                    content = '\n'.join(lines)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except (UnicodeDecodeError, PermissionError) as e:
        print(f"Could not process {file_path}: {e}")
        return False

def main():
    if len(sys.argv) > 1:
        files_to_fix = sys.argv[1:]
    else:
        # Default to scripts directory
        project_root = Path(__file__).parent.parent
        scripts_dir = project_root / 'scripts'
        files_to_fix = []
        for py_file in scripts_dir.rglob('*.py'):
            files_to_fix.append(str(py_file))
    
    print(f"Fixing imports in {len(files_to_fix)} files...")
    
    fixed_count = 0
    for file_path in files_to_fix:
        if fix_file_imports(file_path):
            relative_path = os.path.relpath(file_path)
            print(f"✅ Fixed: {relative_path}")
            fixed_count += 1
        else:
            relative_path = os.path.relpath(file_path)
            print(f"⏭️  No changes: {relative_path}")
    
    print(f"\nFixed {fixed_count} files.")

if __name__ == "__main__":
    main()
