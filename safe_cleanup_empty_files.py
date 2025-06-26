#!/usr/bin/env python3
"""
Safe Empty File Cleanup Script
==============================

This script identifies and removes empty files while preserving:
- Config files (.json, .yaml, .yml, .toml, .ini, .cfg)
- Important infrastructure files (__init__.py, etc.)
- Documentation files
- Version control files
- Build/deployment files
"""

import os
import sys

def is_empty_file(filepath):
    """Check if a file is truly empty (0 bytes)."""
    try:
        return os.path.getsize(filepath) == 0
    except OSError:
        return False

def is_config_file(filepath):
    """Check if a file is a configuration file that should be preserved."""
    config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}
    config_names = {'dockerfile', 'makefile', 'requirements.txt', 'setup.py', 'setup.cfg'}
    
    # Check extension
    _, ext = os.path.splitext(filepath.lower())
    if ext in config_extensions:
        return True
    
    # Check filename
    filename = os.path.basename(filepath.lower())
    if filename in config_names:
        return True
    
    # Check for dotfiles (hidden config files)
    if filename.startswith('.') and not filename.endswith('.py'):
        return True
    
    return False

def is_important_file(filepath):
    """Check if a file is important infrastructure that should be preserved."""
    important_files = {
        '__init__.py',  # Python package files
        'readme.md', 'readme.txt', 'readme.rst',
        'license', 'license.txt', 'license.md',
        'changelog.md', 'changelog.txt',
        'version.py', 'version.txt',
        'manifest.in',
        'pyproject.toml',
        '.gitignore', '.gitkeep',
        'docker-compose.yml', 'docker-compose.yaml',
    }
    
    filename = os.path.basename(filepath.lower())
    return filename in important_files

def should_preserve_file(filepath):
    """Determine if a file should be preserved even if empty."""
    return is_config_file(filepath) or is_important_file(filepath)

def find_safe_empty_files(root_dir):
    """Find empty files that are safe to remove."""
    empty_files = []
    preserved_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip certain directories entirely
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            
            if is_empty_file(filepath):
                if should_preserve_file(filepath):
                    preserved_files.append(filepath)
                else:
                    empty_files.append(filepath)
    
    return empty_files, preserved_files

def main():
    project_root = "/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks"
    
    print("🔍 Scanning for empty files to safely remove...")
    print("✅ Preserving config files, __init__.py, and important infrastructure")
    
    empty_files, preserved_files = find_safe_empty_files(project_root)
    
    print(f"\n📊 Found {len(empty_files)} empty files safe to remove")
    print(f"🛡️  Found {len(preserved_files)} empty files to preserve")
    
    if preserved_files:
        print("\n🛡️  Preserving these empty files:")
        for f in sorted(preserved_files):
            rel_path = os.path.relpath(f, project_root)
            print(f"   ✓ {rel_path}")
    
    if empty_files:
        print(f"\n🗑️  Empty files to remove:")
        
        # Group by directory for better organization
        by_dir = {}
        for f in empty_files:
            rel_path = os.path.relpath(f, project_root)
            dirname = os.path.dirname(rel_path)
            if dirname not in by_dir:
                by_dir[dirname] = []
            by_dir[dirname].append(os.path.basename(rel_path))
        
        for dirname in sorted(by_dir.keys()):
            if dirname == '.':
                print(f"   📁 Root:")
            else:
                print(f"   📁 {dirname}/:")
            for filename in sorted(by_dir[dirname]):
                print(f"      - {filename}")
        
        print(f"\n❓ Remove {len(empty_files)} empty files? (y/N): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            removed_count = 0
            for filepath in empty_files:
                try:
                    os.remove(filepath)
                    removed_count += 1
                except OSError as e:
                    print(f"❌ Could not remove {filepath}: {e}")
            
            print(f"✅ Successfully removed {removed_count} empty files")
        else:
            print("❌ Cleanup cancelled")
    else:
        print("\n✨ No empty files found that are safe to remove!")

if __name__ == "__main__":
    main()
