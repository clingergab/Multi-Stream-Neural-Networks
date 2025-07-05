#!/usr/bin/env python3
"""
Usage example for comprehensive model diagnostics.

This script demonstrates how to run comprehensive diagnostics on 
multi-channel neural networks for CIFAR-100 dataset.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.comprehensive_model_diagnostics import ComprehensiveModelDiagnostics


def main():
    """
    Run comprehensive diagnostics example.
    """
    # Configuration
    config = {
        'data_dir': 'data/cifar-100',
        'output_dir': 'results/comprehensive_diagnostics',
        'device': 'auto',  # Will use MPS on Apple Silicon, CUDA on NVIDIA, or CPU
        'batch_size': 32,  # Reduce to 16 or 8 if running out of memory on MPS
        'epochs': 20,
        'early_stopping_patience': 5
    }
    
    print("🚀 Starting Comprehensive Model Diagnostics")
    print("=" * 50)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Check if data directory exists
    data_path = Path(config['data_dir'])
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("Please ensure CIFAR-100 data is available in the specified directory.")
        return
    
    # Create diagnostics instance
    diagnostics = ComprehensiveModelDiagnostics(
        output_dir=config['output_dir'],
        device=config['device']
    )
    
    # Run comprehensive diagnostics
    try:
        diagnostics.run_comprehensive_diagnostics(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            early_stopping_patience=config['early_stopping_patience']
        )
        
        print("\n✅ Diagnostics completed successfully!")
        print(f"📁 Results saved to: {config['output_dir']}")
        print("\nGenerated files:")
        print("- Training curves and diagnostics plots")
        print("- Gradient flow analysis")
        print("- Weight magnitude analysis")
        print("- Model comparison plots")
        print("- Architecture analysis (JSON)")
        print("- Comprehensive diagnostic report (Markdown)")
        
    except Exception as e:
        print(f"❌ Error running diagnostics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
