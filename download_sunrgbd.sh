#!/bin/bash
# Download SUN RGB-D dataset

echo "=========================================="
echo "SUN RGB-D Dataset Download"
echo "=========================================="

# Create data directory
mkdir -p data/sunrgbd
cd data/sunrgbd

echo ""
echo "Downloading SUN RGB-D dataset (this will take a while - ~37GB)..."
echo ""

# Download main dataset (using curl for macOS compatibility)
echo "Downloading SUNRGBD.zip (~37GB)..."
curl -L http://rgbd.cs.princeton.edu/data/SUNRGBD.zip -o SUNRGBD.zip

# Download toolbox (contains train/test splits and annotations)
echo "Downloading SUNRGBDtoolbox.zip..."
curl -L http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip -o SUNRGBDtoolbox.zip

echo ""
echo "Extracting dataset..."
unzip -q SUNRGBD.zip
unzip -q SUNRGBDtoolbox.zip

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Dataset location: data/sunrgbd/"
echo "  - SUNRGBD/: Main dataset (10,335 RGB-D images)"
echo "  - SUNRGBDtoolbox/: Annotations and splits"
echo ""
echo "Dataset stats:"
echo "  - Total images: 10,335"
echo "  - Train: 5,285 images"
echo "  - Test: 5,050 images"
echo "  - Scene classes: 37"
echo "  - Object classes: 19 (detection) / 700 (segmentation)"
echo ""
echo "Next steps:"
echo "  1. Create SUN RGB-D dataloader"
echo "  2. Update training config for 37 classes"
echo "  3. Train with more data!"
