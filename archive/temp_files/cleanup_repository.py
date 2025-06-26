#!/usr/bin/env python3
"""
Repository Cleanup Script
Organizes the repository by moving temporary files, consolidating documentation,
and preparing for a clean commit.
"""

import os
import shutil
from pathlib import Path

def create_directory_if_not_exists(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def move_file_safely(src, dst):
    """Move file safely, creating destination directory if needed."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    if Path(src).exists():
        if dst_path.exists():
            # Add timestamp to avoid conflicts
            import time
            timestamp = int(time.time())
            dst = f"{dst}.{timestamp}"
        
        shutil.move(src, dst)
        print(f"Moved: {src} -> {dst}")
        return True
    return False

def cleanup_repository():
    """Main cleanup function."""
    print("🧹 Starting repository cleanup...")
    
    # Create cleanup directories
    cleanup_dirs = [
        "archive/development_files",
        "archive/test_scripts", 
        "archive/documentation",
        "archive/old_notebooks"
    ]
    
    for dir_path in cleanup_dirs:
        create_directory_if_not_exists(dir_path)
    
    # Files to move to archive/development_files
    development_files = [
        "architecture_summary.py",
        "bottleneck_analysis.py", 
        "check_optimizer_calc.py",
        "compare_models.py",
        "complete_pipeline_example.py",
        "complete_structure.py",
        "condensation_summary.py",
        "create_final_files.py",
        "create_missing_files.py", 
        "create_remaining_files.py",
        "demo_gpu_optimization.py",
        "demo_multi_channel.py",
        "diagnostic_test.py",
        "download_datasets.py",
        "final_comprehensive_test.py",
        "final_condensation_summary.py",
        "final_document_summary.py",
        "final_refactored_test.py",
        "final_resnet_check.py", 
        "final_verification.py",
        "fix_comparisons.py",
        "gradient_flow_test.py",
        "simple_end_to_end_test.py",
        "train_multi_channel.py",
        "verification_summary.py"
    ]
    
    # Test scripts to move to archive/test_scripts
    test_scripts = [
        "test_actual_components.py",
        "test_actual_refactored_models.py", 
        "test_architecture.py",
        "test_canonical_components.py",
        "test_canonical_dataset_wrapper.py",
        "test_complete_end_to_end.py",
        "test_end_to_end.py",
        "test_gpu_optimization.py",
        "test_multi_channel.py",
        "test_refactored_models_e2e.py",
        "test_refactored_modules.py",
        "test_refactored_transforms.py",
        "test_resnet_mnist.py",
        "test_streamlined_transforms.py",
        "test_training_api.py",
        "test_working_models.py",
        "verify_architecture.py",
        "verify_calculations.py", 
        "verify_corrections.py",
        "verify_equivalence.py",
        "verify_flops.py",
        "verify_gradients.py"
    ]
    
    # Documentation files to move to archive/documentation  
    doc_files = [
        "ARCHITECTURE_REDESIGN.md",
        "CLEANUP_SUMMARY.md",
        "DATA_PIPELINE_CORRECTION.md", 
        "END_TO_END_TESTING_ANALYSIS.md",
        "FINAL_ANALYSIS_REPORT.md",
        "FINAL_COMPLETION_REPORT.md",
        "PROGRESS_SUMMARY.md",
        "PROJECT_STATUS.md", 
        "REFACTORING_COMPLETE.md",
        "REFACTORING_COMPLETED.md",
        "REFACTORING_SUMMARY.md",
        "STREAMLINED_TRANSFORMS_GUIDE.md",
        "TRANSFORM_REFACTORING_ANALYSIS.md",
        "VERIFICATION_SUMMARY.md"
    ]
    
    # Old notebooks to move to archive/old_notebooks
    old_notebooks = [
        "project_structure_guide.ipynb"
    ]
    
    # Move files
    moved_count = 0
    
    print("\n📁 Moving development files...")
    for file in development_files:
        if move_file_safely(file, f"archive/development_files/{file}"):
            moved_count += 1
    
    print("\n🧪 Moving test scripts...")  
    for file in test_scripts:
        if move_file_safely(file, f"archive/test_scripts/{file}"):
            moved_count += 1
    
    print("\n📄 Moving documentation files...")
    for file in doc_files:
        if move_file_safely(file, f"archive/documentation/{file}"):
            moved_count += 1
            
    print("\n📓 Moving old notebooks...")
    for file in old_notebooks:
        if move_file_safely(file, f"archive/old_notebooks/{file}"):
            moved_count += 1
    
    # Keep essential files in root
    essential_files = [
        "README.md",
        "LICENSE", 
        "requirements.txt",
        "setup.py",
        ".gitignore",
        "DESIGN.md",
        "GPU_OPTIMIZATION_SUMMARY.md",
        "GPU_OPTIMIZATION_COMPLETE.md", 
        "READY_FOR_COLAB.md",
        "Multi_Stream_CIFAR100_Training.ipynb",
        "Multi_Stream_NN_Complete_Pipeline.ipynb"
    ]
    
    # Keep essential test and verification scripts
    essential_scripts = [
        "test_basic.py",
        "test_training.py", 
        "test_gpu_optimizations.py",
        "verify_api.py",
        "run_complete_tests.py",
        "run_all_tests.py"
    ]
    
    print(f"\n✅ Moved {moved_count} files to archive directories")
    print("📂 Essential files kept in root:")
    for file in essential_files + essential_scripts:
        if os.path.exists(file):
            print(f"   ✓ {file}")
    
    # Create a summary of what was kept vs moved
    print("\n📊 Cleanup Summary:")
    print(f"   🗂️  Archived: {moved_count} files")
    print(f"   📁 Essential files in root: {len(essential_files + essential_scripts)}")
    print("   📂 Directory structure preserved: src/, data/, examples/, tests/")
    
    return moved_count

if __name__ == "__main__":
    moved = cleanup_repository()
    print(f"\n🎉 Repository cleanup complete! Moved {moved} files to archive.")
