import torch
import random
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Callable, Tuple, List, Dict, Union
from PIL import Image
from .rgb_to_rgbl import RGBtoRGBL

class StreamingDualChannelDataset(Dataset):
    """
    A PyTorch Dataset that provides dual-stream data (RGB + brightness) with consistent augmentation,
    loading ImageNet images from local folders on-the-fly.
    
    Features:
    - Loads ImageNet data on-demand from local folder structure
    - Consistent augmentation across RGB and brightness channels using shared random seeds
    - Standard PyTorch Dataset interface - works with any DataLoader
    - Brightness conversion from RGB on-the-fly
    - Memory efficient - no pre-loading required
    - Handles train (class names in filenames) and val/test (truth file) splits
    - Modular architecture with single-responsibility methods for maintainability
    
    Architecture:
    - Dataset building: _build_dataset(), _build_training_dataset(), _build_validation_test_dataset()
    - Data collection: _collect_image_paths_from_folder(), _collect_all_image_paths()
    - Label handling: _extract_class_from_filename(), _load_labels_from_truth_file(), _create_dummy_labels_for_test()
    - Image loading: _load_and_preprocess_image()
    - Transform handling: _apply_synchronized_transforms(), _create_brightness_compatible_transform()
    """
    
    def __init__(
        self,
        data_folders: Union[str, List[str]],
        split: str = "train",
        truth_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
        valid_extensions: Tuple[str, ...] = ('.JPEG', '.jpg', '.jpeg', '.png')
    ):
        """
        Initialize the streaming dual-channel dataset.
        
        Args:
            data_folders: Path to folder(s) containing images. Can be:
                - Single folder path (str)
                - List of folder paths for multiple train folders
            split: Dataset split ("train", "validation", "test")
            truth_file: Path to truth file for validation/test splits.
                       Expected format: one label per line corresponding to image order
            transform: Transform applied to both RGB and brightness channels with 
                      guaranteed synchronization using the same random seed
            image_size: Target image size (H, W) for resizing
            valid_extensions: Valid image file extensions
        """
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.valid_extensions = valid_extensions
        
        # Create brightness converter
        self._rgb_converter = RGBtoRGBL()
        
        # Handle single folder or multiple folders
        if isinstance(data_folders, str):
            data_folders = [data_folders]
        self.data_folders = [Path(folder) for folder in data_folders]
        
        # Validate folders exist
        for folder in self.data_folders:
            if not folder.exists():
                raise FileNotFoundError(f"Data folder not found: {folder}")
        
        # Build image paths and labels
        self.image_paths, self.labels = self._build_dataset(truth_file)
        
        print(f"✅ Loaded {split} split: {len(self.image_paths)} images from {len(self.data_folders)} folder(s)")
    
    def _build_dataset(self, truth_file: Optional[str]) -> Tuple[List[Path], List[int]]:
        """
        Build the dataset by scanning folders and extracting labels.
        
        Args:
            truth_file: Path to truth file for val/test splits
            
        Returns:
            Tuple of (image_paths, labels)
        """
        if self.split == "train":
            return self._build_training_dataset()
        else:
            return self._build_validation_test_dataset(truth_file)
    
    def _build_training_dataset(self) -> Tuple[List[Path], List[int]]:
        """
        Build training dataset by extracting labels from filenames.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        class_to_idx = {}
        
        # Process all folders to collect paths and build class mapping
        for folder in self.data_folders:
            folder_paths, folder_labels, class_to_idx = self._process_training_folder(
                folder, class_to_idx, len(class_to_idx)
            )
            image_paths.extend(folder_paths)
            labels.extend(folder_labels)
        
        print(f"Found {len(class_to_idx)} classes in training data")
        self.class_to_idx = class_to_idx
        
        return image_paths, labels
    
    def _process_training_folder(
        self, 
        folder: Path, 
        class_to_idx: Dict[str, int], 
        current_idx: int
    ) -> Tuple[List[Path], List[int], Dict[str, int]]:
        """
        Process a single training folder to extract paths and labels.
        
        Args:
            folder: Path to training folder
            class_to_idx: Existing class name to index mapping
            current_idx: Next available class index
            
        Returns:
            Tuple of (folder_image_paths, folder_labels, updated_class_to_idx)
        """
        folder_image_paths = self._collect_image_paths_from_folder(folder)
        valid_paths = []
        folder_labels = []
        
        for img_path in folder_image_paths:
            class_name = self._extract_class_from_filename(img_path)
            if class_name:
                # Map class name to index
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = current_idx
                    current_idx += 1
                
                valid_paths.append(img_path)
                folder_labels.append(class_to_idx[class_name])
        
        return valid_paths, folder_labels, class_to_idx
    
    def _build_validation_test_dataset(self, truth_file: Optional[str]) -> Tuple[List[Path], List[int]]:
        """
        Build validation or test dataset using truth file or dummy labels.
        
        Args:
            truth_file: Path to truth file for labels
            
        Returns:
            Tuple of (image_paths, labels)
        """
        # Collect all image paths from all folders
        image_paths = self._collect_all_image_paths()
        
        # Get labels from truth file or create dummy labels
        labels = self._get_validation_test_labels(truth_file, len(image_paths))
        
        self.class_to_idx = None  # Not needed for val/test
        return image_paths, labels
    
    def _collect_all_image_paths(self) -> List[Path]:
        """
        Collect all image paths from all data folders.
        
        Returns:
            Sorted list of all image paths
        """
        all_image_paths = []
        for folder in self.data_folders:
            folder_image_paths = self._collect_image_paths_from_folder(folder)
            all_image_paths.extend(folder_image_paths)
        return sorted(all_image_paths)
    
    def _get_validation_test_labels(self, truth_file: Optional[str], num_images: int) -> List[int]:
        """
        Get labels for validation/test dataset from truth file or create dummy labels.
        
        Args:
            truth_file: Path to truth file (optional)
            num_images: Number of images in dataset
            
        Returns:
            List of integer labels
        """
        if truth_file is None:
            return self._create_dummy_labels_for_test(num_images)
        else:
            return self._load_labels_from_truth_file(truth_file, num_images)
    
    def _collect_image_paths_from_folder(self, folder: Path) -> List[Path]:
        """
        Collect all valid image paths from a single folder.
        
        Args:
            folder: Path to folder to scan
            
        Returns:
            Sorted list of image paths
        """
        folder_paths = []
        for ext in self.valid_extensions:
            folder_paths.extend(folder.glob(f"*{ext}"))
        return sorted(folder_paths)
    
    def _extract_class_from_filename(self, img_path: Path) -> Optional[str]:
        """
        Extract class name from ImageNet training filename.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Class name if valid format, None otherwise
        """
        filename = img_path.stem
        if '_' in filename:
            return filename.split('_')[0]  # First part before underscore
        return None
    
    def _create_dummy_labels_for_test(self, num_images: int) -> List[int]:
        """
        Create dummy labels for test set when no truth file is provided.
        
        Args:
            num_images: Number of images in dataset
            
        Returns:
            List of dummy labels (all zeros)
        """
        if self.split == "test":
            print(f"Warning: No truth file provided for test split. Using dummy labels.")
            return [0] * num_images
        else:
            raise ValueError(f"truth_file required for {self.split} split")
    
    def _load_labels_from_truth_file(self, truth_file: str, num_images: int) -> List[int]:
        """
        Load labels from truth file and validate count.
        
        Args:
            truth_file: Path to truth file
            num_images: Expected number of images
            
        Returns:
            List of integer labels
            
        Raises:
            FileNotFoundError: If truth file doesn't exist
            ValueError: If label count doesn't match image count
        """
        truth_path = Path(truth_file)
        if not truth_path.exists():
            raise FileNotFoundError(f"Truth file not found: {truth_path}")
        
        # Load truth labels
        with open(truth_path, 'r') as f:
            truth_labels = [int(line.strip()) for line in f]
        
        # Validate we have the right number of labels
        if num_images != len(truth_labels):
            raise ValueError(
                f"Mismatch between images ({num_images}) and truth labels ({len(truth_labels)})"
            )
        
        return truth_labels
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (rgb_tensor, brightness_tensor, label)
        """
        # Load image and label
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Load and preprocess RGB image
        rgb = self._load_and_preprocess_image(img_path)
        
        # Compute brightness from RGB on-the-fly
        brightness = self._rgb_converter.get_brightness(rgb)
        
        # Apply synchronized transforms to both channels
        rgb, brightness = self._apply_synchronized_transforms(rgb, brightness)
        
        return rgb, brightness, label
    
    def _load_and_preprocess_image(self, img_path: Path) -> torch.Tensor:
        """
        Load image from path and convert to RGB tensor.
        
        Args:
            img_path: Path to image file
            
        Returns:
            RGB tensor
            
        Raises:
            RuntimeError: If image loading fails
        """
        try:
            # Load and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Resize to target size
            image = image.resize(self.image_size, Image.LANCZOS)
            
            # Convert to tensor
            rgb = transforms.ToTensor()(image)
            
            return rgb
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
    
    def _apply_synchronized_transforms(
        self, 
        rgb: torch.Tensor, 
        brightness: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms to both RGB and brightness channels with synchronized random states.
        
        Args:
            rgb: RGB tensor
            brightness: Brightness tensor
            
        Returns:
            Tuple of (transformed_rgb, transformed_brightness)
        """
        if not self.transform:
            return rgb, brightness
        
        # Generate seed for consistent augmentation across both channels
        seed = random.randint(0, 2**32 - 1)
        
        # Apply transform to RGB
        rgb = self._apply_transform_with_seed(rgb, self.transform, seed)
        
        # Apply transform to brightness (with fallback for incompatible transforms)
        brightness = self._apply_brightness_transform_with_seed(brightness, seed)
        
        return rgb, brightness
    
    def _apply_transform_with_seed(
        self, 
        tensor: torch.Tensor, 
        transform: Callable, 
        seed: int
    ) -> torch.Tensor:
        """
        Apply transform to tensor with specified random seed.
        
        Args:
            tensor: Input tensor
            transform: Transform to apply
            seed: Random seed for reproducible augmentation
            
        Returns:
            Transformed tensor
        """
        torch.manual_seed(seed)
        random.seed(seed)
        return transform(tensor)
    
    def _apply_brightness_transform_with_seed(
        self, 
        brightness: torch.Tensor, 
        seed: int
    ) -> torch.Tensor:
        """
        Apply transform to brightness channel with fallback for incompatible normalization.
        
        Args:
            brightness: Brightness tensor
            seed: Random seed for reproducible augmentation
            
        Returns:
            Transformed brightness tensor
        """
        try:
            return self._apply_transform_with_seed(brightness, self.transform, seed)
        except RuntimeError as e:
            if "doesn't match the broadcast shape" in str(e):
                # Transform expects 3 channels but brightness has 1
                # Create brightness-compatible version and apply
                brightness_transform = self._create_brightness_compatible_transform(self.transform)
                return self._apply_transform_with_seed(brightness, brightness_transform, seed)
            else:
                raise e
    
    def _create_brightness_compatible_transform(self, original_transform):
        """
        Create a brightness-compatible transform by removing normalization that expects 3 channels.
        
        Args:
            original_transform: Original transform that may contain 3-channel normalization
            
        Returns:
            Transform that works with single-channel brightness
        """
        if not isinstance(original_transform, transforms.Compose):
            return original_transform
        
        compatible_transforms = []
        for t in original_transform.transforms:
            if isinstance(t, transforms.Normalize):
                # Skip normalization for brightness channel or create single-channel version
                if len(t.mean) == 3:
                    # Create single-channel normalization using luminance weights
                    # Standard RGB to luminance: 0.299*R + 0.587*G + 0.114*B
                    luminance_mean = 0.299 * t.mean[0] + 0.587 * t.mean[1] + 0.114 * t.mean[2]
                    luminance_std = 0.299 * t.std[0] + 0.587 * t.std[1] + 0.114 * t.std[2]
                    compatible_transforms.append(transforms.Normalize([luminance_mean], [luminance_std]))
                else:
                    compatible_transforms.append(t)
            else:
                compatible_transforms.append(t)
        
        return transforms.Compose(compatible_transforms)

def create_imagenet_dual_channel_train_val_dataloaders(
    train_folders: Union[str, List[str]],
    val_folder: str,
    truth_file: str,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for ImageNet.
    
    Args:
        train_folders: Path(s) to training folder(s). Can be:
            - Single folder: '/path/to/train_images_0'
            - Multiple folders: ['/path/to/train_images_0', '/path/to/train_images_1', ...]
        val_folder: Path to validation folder
        truth_file: Path to validation truth file (ILSVRC2012_validation_ground_truth.txt)
        train_transform: Transform to apply to training data (both RGB and brightness)
        val_transform: Transform to apply to validation data (both RGB and brightness)
        batch_size: Batch size for training
        val_batch_size: Batch size for validation (defaults to batch_size if not specified)
        image_size: Target image size (H, W) for resizing
        num_workers: Number of parallel data loading workers
        pin_memory: Whether to pin memory for faster CPU→GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    
    # Default validation batch size to training batch size if not specified
    if val_batch_size is None:
        val_batch_size = batch_size
    
    # Create datasets
    train_dataset = StreamingDualChannelDataset(
        data_folders=train_folders,
        split="train",
        truth_file=None,  # Training uses filename labels
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset = StreamingDualChannelDataset(
        data_folders=val_folder,
        split="validation",
        truth_file=truth_file,
        transform=val_transform,
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True  # Drop incomplete batches for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False  # Keep all validation samples
    )
    
    return train_loader, val_loader


def create_imagenet_dual_channel_test_dataloader(
    test_folder: str,
    truth_file: Optional[str] = None,
    test_transform: Optional[Callable] = None,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    Create test dataloader for ImageNet.

    Args:
        test_folder: Path to test folder
        truth_file: Path to test truth file (optional - test may not have labels)
        test_transform: Transform to apply to test data
        batch_size: Batch size for testing
        image_size: Target image size (H, W) for resizing
        num_workers: Number of parallel data loading workers
        pin_memory: Whether to pin memory for faster CPU→GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        Test dataloader
    """
    
    # Create test dataset
    test_dataset = StreamingDualChannelDataset(
        data_folders=test_folder,
        split="test",
        truth_file=truth_file,
        transform=test_transform,
        image_size=image_size
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False  # Keep all test samples
    )

    return test_loader


def create_default_imagenet_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[Callable, Callable]:
    """
    Create default ImageNet transforms for training and validation.
    
    Args:
        image_size: Target image size (H, W)
        mean: ImageNet normalization mean for RGB channels
        std: ImageNet normalization std for RGB channels
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.143)),  # Resize to 256 for 224x224
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform