# Data Directory

This directory will contain datasets and processed data for Multi-Stream Neural Networks.

## Structure
```
data/
├── raw/                    # Original datasets
│   ├── cifar10/
│   ├── imagenet/
│   └── custom/
├── processed/              # Preprocessed data
│   ├── color_pathways/
│   ├── brightness_pathways/
│   └── combined/
├── splits/                 # Train/val/test splits
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── cache/                  # Cached preprocessed data
    ├── features/
    └── statistics/
```

## Usage

The data loading pipeline will:
1. Download raw datasets to `raw/`
2. Preprocess and separate color/brightness channels
3. Store processed data in `processed/`
4. Cache frequently used data in `cache/`

## Datasets Supported

- **CIFAR-10**: Color classification benchmark
- **ImageNet**: Large-scale image classification
- **Custom**: User-provided datasets

## Color/Brightness Separation

The preprocessing pipeline separates input images into:
- **Color pathway**: RGB color information
- **Brightness pathway**: Luminance/grayscale information

This separation mimics the biological distinction between parvocellular (color) and magnocellular (brightness) pathways in human vision.
