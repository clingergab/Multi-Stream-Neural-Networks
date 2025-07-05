#!/usr/bin/env python3
"""
Repository Cleanup Script

This script helps clean up the repository by:
1. Removing large files that are now in .gitignore
2. Creating necessary placeholder README files
3. Organizing the repository structure
4. Generating data download scripts instead of storing large datasets
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

def run_command(command):
    """Run a shell command and return the output"""
    process = subprocess.run(
        command, 
        shell=True, 
        text=True, 
        capture_output=True
    )
    return process.stdout.strip()

def is_git_tracked(file_path):
    """Check if a file is tracked by Git"""
    result = run_command(f"git ls-files --error-unmatch {file_path} 2>/dev/null || echo 'untracked'")
    return result != 'untracked'

def create_directory_with_readme(directory, content):
    """Create a directory with a README.md file"""
    os.makedirs(directory, exist_ok=True)
    
    readme_path = os.path.join(directory, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(content)
        print(f"Created README.md in {directory}")

def create_data_download_script():
    """Create scripts to download datasets instead of storing them"""
    scripts_dir = "scripts/data"
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Create MNIST download script
    mnist_script = os.path.join(scripts_dir, "download_mnist.py")
    with open(mnist_script, "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
MNIST Dataset Downloader
\"\"\"
import os
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

def download_mnist(data_dir='data/MNIST'):
    \"\"\"Download MNIST dataset\"\"\"
    print(f"Downloading MNIST dataset to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download training data
    MNIST(root=data_dir, train=True, download=True, 
          transform=transforms.ToTensor())
    
    # Download test data
    MNIST(root=data_dir, train=False, download=True,
          transform=transforms.ToTensor())
    
    print("MNIST dataset downloaded successfully!")

if __name__ == "__main__":
    download_mnist()
""")
    
    # Create CIFAR-100 download script
    cifar_script = os.path.join(scripts_dir, "download_cifar100.py")
    with open(cifar_script, "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
CIFAR-100 Dataset Downloader
\"\"\"
import os
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms

def download_cifar100(data_dir='data/cifar-100'):
    \"\"\"Download CIFAR-100 dataset\"\"\"
    print(f"Downloading CIFAR-100 dataset to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download training data
    CIFAR100(root=data_dir, train=True, download=True,
            transform=transforms.ToTensor())
    
    # Download test data
    CIFAR100(root=data_dir, train=False, download=True,
            transform=transforms.ToTensor())
    
    print("CIFAR-100 dataset downloaded successfully!")

if __name__ == "__main__":
    download_cifar100()
""")
    
    # Create combined download script
    all_script = os.path.join(scripts_dir, "download_all.py")
    with open(all_script, "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Download All Datasets
\"\"\"
from download_mnist import download_mnist
from download_cifar100 import download_cifar100

if __name__ == "__main__":
    print("Downloading all required datasets...")
    download_mnist()
    download_cifar100()
    print("All datasets downloaded successfully!")
""")

    # Make scripts executable
    for script in [mnist_script, cifar_script, all_script]:
        os.chmod(script, 0o755)
    
    print(f"Created dataset download scripts in {scripts_dir}/")
    print(f"You can run: python {scripts_dir}/download_all.py to download all datasets")

def remove_large_files_from_git():
    """Remove large files from Git tracking but keep them locally"""
    large_file_patterns = [
        "saved_models/*.pth",
        "saved_models/*.pt",
        "data/MNIST/*",
        "data/cifar-100/*",
        "data/cifar-100-python/*",
        "data/ImageNet/*",
        "checkpoints/*",
    ]
    
    for pattern in large_file_patterns:
        print(f"Checking for files matching: {pattern}")
        files = run_command(f"git ls-files {pattern} 2>/dev/null || echo ''").split("\n")
        files = [f for f in files if f.strip()]
        
        for file in files:
            if file and os.path.exists(file):
                # Remove from git but keep the file
                print(f"Removing {file} from Git tracking (keeping the file locally)")
                run_command(f"git rm --cached '{file}'")

def create_directory_structure():
    """Create a clean directory structure with README files"""
    directories = {
        "data": "# Data Directory\n\nThis directory contains datasets used by the project.\n\nLarge datasets are not stored in the repository. Use the download scripts in `scripts/data/` to download them.",
        "saved_models": "# Saved Models\n\nThis directory contains trained model checkpoints.\n\nLarge model files are not stored in the repository. They should be stored separately.",
        "checkpoints": "# Checkpoints Directory\n\nThis directory contains training checkpoints.\n\nLarge checkpoint files are not stored in the repository.",
        "results": "# Results Directory\n\nThis directory contains experiment results and visualizations.",
        "scripts/data": "# Data Scripts\n\nThis directory contains scripts for downloading and preparing datasets.",
        "scripts/training": "# Training Scripts\n\nThis directory contains scripts for training models.",
        "scripts/evaluation": "# Evaluation Scripts\n\nThis directory contains scripts for evaluating models.",
        "src/data": "# Data Processing\n\nThis directory contains data processing utilities.",
        "src/models": "# Model Definitions\n\nThis directory contains model architecture definitions.",
        "src/utils": "# Utilities\n\nThis directory contains utility functions used throughout the project."
    }
    
    for directory, content in directories.items():
        create_directory_with_readme(directory, content)

def main():
    parser = argparse.ArgumentParser(description="Clean up the repository")
    parser.add_argument("--remove-git", action="store_true", help="Remove large files from Git tracking")
    parser.add_argument("--create-structure", action="store_true", help="Create directory structure with READMEs")
    parser.add_argument("--create-scripts", action="store_true", help="Create data download scripts")
    parser.add_argument("--all", action="store_true", help="Run all cleanup operations")
    
    args = parser.parse_args()
    
    # If no args specified, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    if args.all or args.create_structure:
        print("\n=== Creating Directory Structure ===")
        create_directory_structure()
    
    if args.all or args.remove_git:
        print("\n=== Removing Large Files from Git ===")
        remove_large_files_from_git()
    
    if args.all or args.create_scripts:
        print("\n=== Creating Data Download Scripts ===")
        create_data_download_script()
    
    print("\n=== Cleanup Complete ===")
    print("""
Next steps:
1. Commit the changes:
   git commit -m "Clean up repository structure"

2. Create a clean repository:
   - Create a new repository on GitHub
   - Push this clean repository to the new GitHub repository
   - Or use the export function in the notebook to create a clean zip file
""")

if __name__ == "__main__":
    main()
