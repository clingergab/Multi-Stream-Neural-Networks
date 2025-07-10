#!/usr/bin/env python3
"""
Script to check for remaining 'from src.' imports in the codebase.
"""

import os
import re
import sys
from pathlib import Path

def find_src_imports(directory):
    """Find all files with 'from src.' imports."""
    src_import_pattern = re.compile(r'from src\.')
    results = []
    
    for root, dirs, files in os.walk(directory):
        # Skip some directories that are not critical
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'htmlcov', '.tox']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        if src_import_pattern.search(line):
                            results.append({
                                'file': file_path,
                                'line': line_num,
                                'content': line.strip()
                            })
                except (UnicodeDecodeError, PermissionError):
                    continue
    
    return results

def categorize_files(results):
    """Categorize files by type."""
    categories = {
        'scripts': [],
        'tests': [],
        'notebooks': [],
        'src': [],
        'other': []
    }
    
    for result in results:
        file_path = result['file']
        if '/scripts/' in file_path:
            categories['scripts'].append(result)
        elif '/tests/' in file_path:
            categories['tests'].append(result)
        elif '/notebooks/' in file_path or file_path.endswith('.ipynb'):
            categories['notebooks'].append(result)
        elif '/src/' in file_path:
            categories['src'].append(result)
        else:
            categories['other'].append(result)
    
    return categories

def main():
    project_root = Path(__file__).parent.parent
    print(f"Checking for 'from src.' imports in: {project_root}")
    print("=" * 80)
    
    results = find_src_imports(project_root)
    
    if not results:
        print("✅ No 'from src.' imports found! All imports have been fixed.")
        return
    
    categories = categorize_files(results)
    
    print(f"Found {len(results)} files with 'from src.' imports")
    print()
    
    # Show summary by category
    for category, items in categories.items():
        if items:
            print(f"{category.upper()}: {len(items)} imports in {len(set(item['file'] for item in items))} files")
    
    print()
    print("PRIORITY ORDER FOR FIXING:")
    print("1. Scripts (most critical for running)")
    print("2. Source files (core functionality)")  
    print("3. Tests (development and CI)")
    print("4. Notebooks (documentation/examples)")
    print()
    
    # Show details for each category
    for category in ['scripts', 'src', 'tests', 'notebooks', 'other']:
        items = categories[category]
        if not items:
            continue
            
        print(f"\n{category.upper()} FILES:")
        print("-" * 50)
        
        files_dict = {}
        for item in items:
            file_path = item['file']
            if file_path not in files_dict:
                files_dict[file_path] = []
            files_dict[file_path].append(item)
        
        for file_path, imports in files_dict.items():
            relative_path = os.path.relpath(file_path, project_root)
            print(f"\n📄 {relative_path} ({len(imports)} imports)")
            for imp in imports[:3]:  # Show first 3 imports
                print(f"   Line {imp['line']}: {imp['content']}")
            if len(imports) > 3:
                print(f"   ... and {len(imports) - 3} more")

if __name__ == "__main__":
    main()
