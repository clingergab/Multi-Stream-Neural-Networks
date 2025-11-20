
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def test_mode_f():
    print("Testing Mode F behavior...")
    
    # Create float array 0..65535
    arr = np.array([[0, 32768], [65535, 10000]], dtype=np.float32)
    img = Image.fromarray(arr, mode='F')
    
    print(f"Original: {arr}")
    
    # To Tensor
    t = transforms.functional.to_tensor(img)
    
    print(f"Tensor shape: {t.shape}")
    print(f"Tensor values: {t}")
    
    if t.max() > 1.0:
        print("Mode F is NOT scaled by to_tensor (Good)")
    else:
        print("Mode F IS scaled by to_tensor (Bad for us)")

if __name__ == "__main__":
    test_mode_f()
