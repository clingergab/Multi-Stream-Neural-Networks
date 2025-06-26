#!/usr/bin/env python3
"""
Comprehensive directory cleanup script.
Moves files to appropriate directories and removes temporary files.
"""

import os
import shutil

def cleanup_directory():
    """Clean up the main directory by organizing files properly."""
    
    print("🧹 Starting comprehensive directory cleanup...")
    
    # Essential files that must stay in root
    keep_in_root = {
        'README.md', 'setup.py', 'requirements.txt', 'LICENSE', '.gitignore',
        'DESIGN.md'  # Core project files only
    }
    
    # Create necessary directories
    directories = {
        'archive/temp_files': 'Temporary files and old scripts',
        'tests': 'Test files', 
        'verification': 'Verification scripts',
        'notebooks': 'Jupyter notebooks',
        'docs/reports': 'Analysis reports and summaries'
    }
    
    for dir_path, description in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created/verified: {dir_path}")
    
    moved_count = 0
    removed_count = 0
    
    # Get all files in root directory
    root_files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('.')]
    
    for file in root_files:
        if file in keep_in_root:
            print(f"✅ Keeping in root: {file}")
            continue
            
        try:
            # Determine destination based on file patterns
            if file.endswith('.ipynb'):
                destination = 'notebooks/'
            elif file.startswith('test_') or file.endswith('_test.py'):
                destination = 'tests/'
            elif file.startswith('verify_') or 'verification' in file.lower():
                destination = 'verification/'
            elif any(word in file.upper() for word in ['SUMMARY', 'ANALYSIS', 'REPORT', 'COMPLETE', 'PROGRESS', 'STATUS']):
                destination = 'docs/reports/'
            elif file.endswith('.py') and any(word in file for word in ['temp', 'final', 'demo', 'check', 'fix', 'create', 'compare', 'bottleneck', 'gradient', 'diagnostic', 'simple', 'complete', 'architecture', 'condensation', 'download', 'train']):
                destination = 'archive/temp_files/'
            elif file.endswith('.md') and file not in keep_in_root:
                destination = 'docs/reports/'
            else:
                # Default: move to archive
                destination = 'archive/temp_files/'
            
            # Move the file
            shutil.move(file, destination + file)
            print(f"📦 Moved {file} → {destination}")
            moved_count += 1
            
        except Exception as e:
            print(f"⚠️  Error moving {file}: {e}")
    
    # Remove any __pycache__ directories
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            shutil.rmtree(cache_dir)
            print(f"🗑️  Removed cache: {cache_dir}")
            removed_count += 1
    
    print("\n✅ Cleanup complete!")
    print(f"   📦 Moved {moved_count} files")
    print(f"   🗑️  Removed {removed_count} cache directories")
    
    # Show final clean root directory
    print("\n📁 Clean root directory contents:")
    root_files = sorted([f for f in os.listdir('.') if os.path.isfile(f)])
    for file in root_files:
        print(f"   ✅ {file}")
    
    # Show directory structure
    print("\n📊 Final directory structure:")
    for item in sorted(os.listdir('.')):
        if os.path.isdir(item) and not item.startswith('.'):
            file_count = len([f for f in os.listdir(item) if os.path.isfile(os.path.join(item, f))])
            print(f"   📁 {item}/ ({file_count} files)")

if __name__ == "__main__":
    cleanup_directory()
