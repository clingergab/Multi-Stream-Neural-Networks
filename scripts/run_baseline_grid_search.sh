#!/bin/bash

# MCResNet Baseline Grid Search Runner
# This script runs the baseline hyperparameter grid search for MCResNet

echo "MCResNet Baseline Grid Search"
echo "============================="
echo "64 combinations x 15 epochs = 960 total epochs"
echo "Estimated time: 5-8 hours"
echo ""

# Ensure we're in the project directory
cd "$(dirname "$0")/.."

# Create results directory
mkdir -p mcresnet_baseline_grid_search

# Check if required modules can be imported
echo "Checking dependencies..."
python -c "
import sys
from pathlib import Path
sys.path.append(str(Path('.') / 'src'))
try:
    from src.models2.multi_channel.mc_resnet import MCResNet
    from src.data_utils.streaming_dual_channel_dataset import create_streaming_dual_channel_train_val_dataloaders
    print('✅ All dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed. Please ensure all required modules are available."
    exit 1
fi

echo ""
echo "Starting grid search..."
echo "Results will be saved to: mcresnet_baseline_grid_search/"
echo ""

# Run the grid search
python scripts/grid_search_mcresnet_baseline.py

echo ""
echo "Grid search completed!"
echo "Check the mcresnet_baseline_grid_search/ directory for results."
