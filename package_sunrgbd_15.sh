#!/bin/bash
# Package SUN RGB-D 15-category dataset for upload to Google Drive

set -e

DATASET_DIR="data/sunrgbd_15"
OUTPUT_FILE="sunrgbd_15_preprocessed.tar.gz"

echo "Packaging SUN RGB-D 15-category dataset..."
echo "Dataset directory: $DATASET_DIR"
echo "Output file: $OUTPUT_FILE"

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    echo "Please run preprocess_sunrgbd_15.py first"
    exit 1
fi

# Create tarball with compression
echo ""
echo "Creating compressed archive..."
tar -czf "$OUTPUT_FILE" \
    --exclude='*.log' \
    --exclude='__pycache__' \
    -C data \
    sunrgbd_15

# Get file size
SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "✓ Package created successfully!"
echo "File: $OUTPUT_FILE"
echo "Size: $SIZE"
echo ""
echo "Next steps:"
echo "1. Upload $OUTPUT_FILE to your Google Drive"
echo "2. In Colab, mount Google Drive and extract:"
echo "   !tar -xzf /content/drive/MyDrive/$OUTPUT_FILE -C /content/data/"
echo ""
echo "Or download directly to Colab from a URL (if you have a public link):"
echo "   !wget YOUR_URL -O sunrgbd_15_preprocessed.tar.gz"
echo "   !mkdir -p /content/data"
echo "   !tar -xzf sunrgbd_15_preprocessed.tar.gz -C /content/data/"
