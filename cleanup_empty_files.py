#!/usr/bin/env python3
"""
Empty File Cleanup Script
========================

This script identifies and removes empty files that are no longer needed,
while preserving important empty files like __init__.py files.
"""

import os
import glob
from pathlib import Path

def find_empty_files(directory):
    """Find all empty files in a directory."""
    empty_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                if os.path.getsize(filepath) == 0:
                    empty_files.append(filepath)
            except (OSError, IOError):
                continue
    return empty_files

def is_important_empty_file(filepath):
    """Check if an empty file should be preserved."""
    filename = os.path.basename(filepath)
    
    # Preserve __init__.py files (they can be empty and still important)
    if filename == '__init__.py':
        return True
    
    # Preserve .gitkeep files (used to keep empty directories in git)
    if filename == '.gitkeep':
        return True
    
    # Preserve any config files that might be intentionally empty
    if filename in ['.gitignore', 'requirements.txt', 'setup.cfg']:
        return True
    
    return False

def categorize_empty_files(empty_files):
    """Categorize empty files for safe removal."""
    archive_files = []
    test_files = []
    src_files = []
    other_files = []
    preserved_files = []
    
    for filepath in empty_files:
        if is_important_empty_file(filepath):
            preserved_files.append(filepath)
        elif '/archive/' in filepath:
            archive_files.append(filepath)
        elif '/tests/' in filepath:
            test_files.append(filepath)
        elif '/src/' in filepath:
            src_files.append(filepath)
        else:
            other_files.append(filepath)
    
    return {
        'archive': archive_files,
        'tests': test_files,
        'src': src_files,
        'other': other_files,
        'preserved': preserved_files
    }

def main():
    """Main cleanup function."""
    project_root = "/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks"
    
    print("🧹 EMPTY FILE CLEANUP")
    print("=" * 50)
    
    # Find all empty files
    print("🔍 Scanning for empty files...")
    empty_files = find_empty_files(project_root)
    
    if not empty_files:
        print("✅ No empty files found!")
        return
    
    print(f"📁 Found {len(empty_files)} empty files")
    
    # Categorize files
    categories = categorize_empty_files(empty_files)
    
    # Report findings
    print("\n📊 EMPTY FILES ANALYSIS:")
    print(f"  🗄️  Archive files: {len(categories['archive'])}")
    print(f"  🧪 Test files: {len(categories['tests'])}")
    print(f"  📦 Source files: {len(categories['src'])}")
    print(f"  📄 Other files: {len(categories['other'])}")
    print(f"  🔒 Preserved files: {len(categories['preserved'])}")
    
    # Remove archive files (safe to remove)
    if categories['archive']:
        print(f"\n🗑️  Removing {len(categories['archive'])} empty archive files...")
        for filepath in categories['archive']:
            try:
                os.remove(filepath)
                print(f"  ✅ Removed: {os.path.relpath(filepath, project_root)}")
            except OSError as e:
                print(f"  ❌ Failed to remove {filepath}: {e}")
    
    # Remove empty test files (likely safe)
    if categories['tests']:
        print(f"\n🧪 Removing {len(categories['tests'])} empty test files...")
        for filepath in categories['tests']:
            try:
                os.remove(filepath)
                print(f"  ✅ Removed: {os.path.relpath(filepath, project_root)}")
            except OSError as e:
                print(f"  ❌ Failed to remove {filepath}: {e}")
    
    # Be more careful with src files
    if categories['src']:
        print(f"\n⚠️  Found {len(categories['src'])} empty source files:")
        for filepath in categories['src']:
            rel_path = os.path.relpath(filepath, project_root)
            print(f"  📄 {rel_path}")
        
        print("\n🤔 Removing empty source files (excluding important ones)...")
        for filepath in categories['src']:
            filename = os.path.basename(filepath)
            # Be extra careful with source files
            if not is_important_empty_file(filepath):
                try:
                    os.remove(filepath)
                    print(f"  ✅ Removed: {os.path.relpath(filepath, project_root)}")
                except OSError as e:
                    print(f"  ❌ Failed to remove {filepath}: {e}")
    
    # Report other files but don't auto-remove
    if categories['other']:
        print(f"\n📋 Found {len(categories['other'])} other empty files (manual review needed):")
        for filepath in categories['other']:
            rel_path = os.path.relpath(filepath, project_root)
            print(f"  📄 {rel_path}")
    
    # Report preserved files
    if categories['preserved']:
        print(f"\n🔒 Preserved {len(categories['preserved'])} important empty files:")
        for filepath in categories['preserved']:
            rel_path = os.path.relpath(filepath, project_root)
            print(f"  📄 {rel_path}")
    
    print("\n✅ Empty file cleanup completed!")
    
    # Final count
    remaining_empty = find_empty_files(project_root)
    print(f"📊 Remaining empty files: {len(remaining_empty)}")

if __name__ == "__main__":
    main()
