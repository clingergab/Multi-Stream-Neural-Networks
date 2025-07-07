#!/usr/bin/env python3
"""
Flexible coverage runner for any test file or folder.
Usage: python3 run_coverage.py <test_file_or_folder> [source_path]
"""

import subprocess
import sys
import os

def infer_source_path(test_path):
    """Infer source path from test path using smart logic."""
    if test_path.endswith('.py'):
        # Single test file - derive source directory from test file path
        test_dir = os.path.dirname(test_path)
        
        # Get the source directory by removing 'tests/' prefix
        if test_dir.startswith('tests/'):
            source_path = test_dir[6:]  # Remove 'tests/' prefix
        else:
            # Fallback to generic src if not following standard pattern
            source_path = "src"
    else:
        # Test folder - derive source folder
        if test_path.startswith('tests/'):
            source_path = test_path[6:]  # Remove 'tests/' prefix
        else:
            source_path = test_path
    
    return source_path

def get_target_file_for_display(test_path):
    """Get the target source file name for display purposes."""
    if test_path.endswith('.py'):
        test_filename = os.path.basename(test_path)
        if test_filename.startswith('test_'):
            # Remove 'test_' prefix to get source filename
            return test_filename[5:]  # Remove 'test_'
    return None

def run_coverage(test_path, source_path=None):
    """Run coverage analysis and generate reports."""
    # If source_path not provided, infer it from test_path
    if source_path is None:
        source_path = infer_source_path(test_path)
    
    # Get target file for display purposes
    target_file = get_target_file_for_display(test_path)
    
    print(f"🔍 Running coverage analysis on: {test_path}")
    print(f"📁 Source path: {source_path}")
    if target_file:
        print(f"🎯 Target file: {target_file}")
    print("="*60)
    
    # Run coverage using pytest-cov (more reliable than coverage run)
    print("Running tests with coverage...")
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        test_path,
        f'--cov={source_path}',
        '--cov-report=term-missing',
        '--cov-report=html',
        '-v'
    ])
    
    if result.returncode != 0:
        print("❌ Tests failed!")
        return False
    
    print("\n" + "="*60)
    print("✅ Coverage analysis completed successfully!")
    print("📄 HTML coverage report generated: htmlcov/index.html")
    print("💡 Open htmlcov/index.html in your browser to view detailed coverage")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_coverage.py <test_file_or_folder> [source_path]")
        print("Examples:")
        print("  # Single test file (auto-detects source file):")
        print("  python3 run_coverage.py tests/src/models2/core/test_resnet.py")
        print("  # -> Analyzes coverage for src/models2/core/resnet.py")
        print("  # Test folder (auto-detects source folder):")
        print("  python3 run_coverage.py tests/src/models2/core")
        print("  # -> Analyzes coverage for src/models2/core/")
        print("  # Explicit source path:")
        print("  python3 run_coverage.py tests/src/models2/core src/models2")
        print("  python3 run_coverage.py tests/src src")
        sys.exit(1)
    
    test_path = sys.argv[1]
    source_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = run_coverage(test_path, source_path)
    sys.exit(0 if success else 1)
