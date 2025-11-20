
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

def test_resize_i16():
    print("Testing I;16 resize behavior...")
    
    # Create a dummy I;16 image (e.g. 100x100)
    # Fill with a gradient to see what happens
    arr = np.linspace(0, 65535, 100*100).reshape(100, 100).astype(np.uint16)
    img = Image.fromarray(arr, mode='I;16')
    
    print(f"Original size: {img.size}, Mode: {img.mode}")
    
    target_size = (200, 200)
    
    # Try resize
    try:
        resized_img = transforms.functional.resize(img, target_size)
        print(f"Resized size: {resized_img.size}, Mode: {resized_img.mode}")
        
        # Check content
        resized_arr = np.array(resized_img)
        print(f"Resized array shape: {resized_arr.shape}")
        print(f"Resized array max: {resized_arr.max()}")
        
        # Check if it's padded (lots of zeros)
        zeros = np.sum(resized_arr == 0)
        print(f"Zeros count: {zeros} / {resized_arr.size}")
        
        if zeros > resized_arr.size * 0.5:
            print("WARNING: Result seems to be padded with zeros!")
            
    except Exception as e:
        print(f"Resize failed: {e}")

    # Try resized_crop
    try:
        # Crop center
        resized_crop_img = transforms.functional.resized_crop(img, 25, 25, 50, 50, target_size)
        print(f"ResizedCrop size: {resized_crop_img.size}, Mode: {resized_crop_img.mode}")
        
        resized_crop_arr = np.array(resized_crop_img)
        zeros = np.sum(resized_crop_arr == 0)
        print(f"Zeros count: {zeros} / {resized_crop_arr.size}")
        
    except Exception as e:
        print(f"ResizedCrop failed: {e}")

if __name__ == "__main__":
    test_resize_i16()
