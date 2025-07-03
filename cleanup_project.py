#!/usr/bin/env python3
"""
Project Cleanup Script
Identifies and removes unnecessary test files, old model files, and other cleanup targets.
"""

import os
import sys
from pathlib import Path

def get_file_age_days(filepath):
    """Get the age of a file in days."""
    import time
    return (time.time() - os.path.getmtime(filepath)) / (24 * 3600)

def analyze_cleanup_candidates():
    """Analyze files that might need cleanup."""
    project_root = Path("/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks")
    
    # Categories of files to analyze
    cleanup_candidates = {
        'old_test_files': [],
        'model_files': [],
        'temp_files': [],
        'log_files': [],
        'verification_files': [],
        'keep_files': []
    }
    
    # Test files that appear to be outdated or one-off tests
    old_test_patterns = [
        'autocast_test.py',
        'mps_test.py',              # Test for Apple Silicon GPU support
        'simple_test.py',           # Standalone test that's been integrated
        'test_all_methods.py',      # Comprehensive test that's been integrated
        'test_api_consistency.py',
        'test_autocast.py', 
        'test_base_refactoring.py',
        'test_batch_size.py',
        'test_cleaned_model.py',
        'test_dataloader_refactor.py',
        'test_grad_scaler_fix.py',
        'test_gradient_clip_defaults.py',
        'test_gradient_clipping.py',
        'test_logging.py',
        'test_method_signatures.py',
        'test_model_defaults.py',
        'test_pathway_analysis.py',  # Debug script for pathway analysis
        'test_refactored_base.py',
        'test_refactoring.py',       # Temporary test file
        'test_shared_classifier_removal.py',
        'test_verbose_placement.py',
        'test_training.py'           # Temporary test file
    ]
    
    # Model files (saved models that might be outdated)
    model_patterns = [
        'base_multichannelnetwork_model.pth',
        'best_basemultichannelnetwork_model.pth',
        'best_multichannelresnetnetwork_model.pth',
        'fixed_multichannelresnetnetwork_model.pth',
        'full_multichannelresnetnetwork_model.pth'
    ]
    
    # Verification files (one-off verification scripts)
    verification_patterns = [
        'final_api_verification.py',
        'final_refactoring_test.py',
        'simple_api_verification.py',
        'verify_api_unification.py',
        'verify_autocast_fix.py',
        'verify_torch_amp.py'
    ]
    
    # Log and diagnostic files
    log_patterns = [
        'diagnostic_results.log'
    ]
    
    # Important files to keep (should NOT be deleted)
    keep_patterns = [
        'tests/',  # Organized test directory
        'src/',    # Source code
        'docs/',   # Documentation
        'configs/', # Configuration files
        'data/',   # Data directory
        'scripts/', # Utility scripts
        'examples/', # Example code
        'notebooks/', # Jupyter notebooks
        'results/', # Results and outputs
        'saved_models/', # Organized model storage
        'README.md',
        'LICENSE',
        'requirements.txt',
        'setup.py',
        '.gitignore',
        'DESIGN.md',
        'recommendations.md',
        'fit_methods_comparison.md',
        '.copilot-instructions.md'
    ]
    
    # Scan for files
    for pattern in old_test_patterns:
        filepath = project_root / pattern
        if filepath.exists():
            cleanup_candidates['old_test_files'].append(str(filepath))
    
    for pattern in model_patterns:
        filepath = project_root / pattern
        if filepath.exists():
            cleanup_candidates['model_files'].append(str(filepath))
    
    for pattern in verification_patterns:
        filepath = project_root / pattern
        if filepath.exists():
            cleanup_candidates['verification_files'].append(str(filepath))
    
    for pattern in log_patterns:
        filepath = project_root / pattern
        if filepath.exists():
            cleanup_candidates['log_files'].append(str(filepath))
    
    # Check for PNG files (might be outdated comparison plots)
    png_files = list(project_root.glob("*.png"))
    for png_file in png_files:
        if get_file_age_days(png_file) > 7:  # Older than a week
            cleanup_candidates['temp_files'].append(str(png_file))
    
    return cleanup_candidates

def print_cleanup_analysis(candidates):
    """Print analysis of cleanup candidates."""
    print("🧹 Project Cleanup Analysis")
    print("=" * 60)
    
    total_files = sum(len(files) for files in candidates.values())
    if total_files == 0:
        print("✅ No cleanup candidates found! Project is already clean.")
        return
    
    for category, files in candidates.items():
        if files:
            print(f"\n📁 {category.replace('_', ' ').title()} ({len(files)} files):")
            for file in files:
                filename = os.path.basename(file)
                age_days = get_file_age_days(file)
                size_kb = os.path.getsize(file) / 1024
                print(f"   📄 {filename} ({size_kb:.1f}KB, {age_days:.1f} days old)")
    
    print(f"\n📊 Total cleanup candidates: {total_files} files")

def perform_cleanup(candidates, dry_run=True):
    """Perform the actual cleanup."""
    print(f"\n{'🔍 DRY RUN' if dry_run else '🗑️  CLEANING UP'} - {'Simulating' if dry_run else 'Executing'} cleanup...")
    
    files_to_remove = []
    for category, files in candidates.items():
        if category != 'keep_files':  # Don't remove keep_files
            files_to_remove.extend(files)
    
    if not files_to_remove:
        print("✅ No files to remove.")
        return
    
    for filepath in files_to_remove:
        try:
            if dry_run:
                print(f"   Would remove: {os.path.basename(filepath)}")
            else:
                os.remove(filepath)
                print(f"   ✅ Removed: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"   ❌ Error with {os.path.basename(filepath)}: {e}")
    
    print(f"\n{'📋 Summary' if dry_run else '✅ Cleanup Complete'}: {len(files_to_remove)} files {'would be' if dry_run else 'were'} removed")

def main():
    """Main cleanup function."""
    print("🚀 Multi-Stream Neural Networks Project Cleanup")
    print("=" * 60)
    
    # Analyze cleanup candidates
    candidates = analyze_cleanup_candidates()
    
    # Print analysis
    print_cleanup_analysis(candidates)
    
    if sum(len(files) for files in candidates.values()) == 0:
        return
    
    # Ask user for confirmation
    print("\n" + "=" * 60)
    print("🤔 Cleanup Options:")
    print("1. Dry run (show what would be removed)")
    print("2. Perform cleanup (actually remove files)")
    print("3. Cancel")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        perform_cleanup(candidates, dry_run=True)
    elif choice == '2':
        print("\n⚠️  Are you sure you want to remove these files? This cannot be undone!")
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm == 'yes':
            perform_cleanup(candidates, dry_run=False)
        else:
            print("❌ Cleanup cancelled.")
    else:
        print("❌ Cleanup cancelled.")

if __name__ == "__main__":
    main()
