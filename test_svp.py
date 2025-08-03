import torch
import torch.nn.functional as F
import time


def test_sequential_vs_parallel_mac():
    """Mac-compatible benchmark script."""
    # Check device availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        supports_streams = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        supports_streams = False  # MPS doesn't support CUDA streams
    else:
        device = torch.device('cpu')
        supports_streams = False
    
    print(f"Testing on: {device}")
    print(f"CUDA Streams supported: {supports_streams}")
    
    # Test data
    batch_size, height, width = 32, 224, 224
    color_input = torch.randn(batch_size, 3, height, width, device=device)
    brightness_input = torch.randn(batch_size, 1, height, width, device=device)
    color_weight = torch.randn(64, 3, 7, 7, device=device)
    brightness_weight = torch.randn(64, 1, 7, 7, device=device)
    
    # Warm up
    for _ in range(10):
        _ = F.conv2d(color_input, color_weight, stride=2, padding=3)
        _ = F.conv2d(brightness_input, brightness_weight, stride=2, padding=3)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Test 1: Sequential
    print("\n=== Sequential Execution ===")
    start = time.perf_counter()
    for _ in range(100):
        color_out = F.conv2d(color_input, color_weight, stride=2, padding=3)
        brightness_out = F.conv2d(brightness_input, brightness_weight, stride=2, padding=3)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    sequential_time = time.perf_counter() - start
    print(f"Sequential: {sequential_time*10:.2f}ms per call")
    
    # Test 2: CUDA Streams (only if supported)
    if supports_streams:
        print("\n=== CUDA Streams ===")
        start = time.perf_counter()
        for _ in range(100):
            color_stream = torch.cuda.Stream()
            brightness_stream = torch.cuda.Stream()
            
            with torch.cuda.stream(color_stream):
                color_out = F.conv2d(color_input, color_weight, stride=2, padding=3)
            
            with torch.cuda.stream(brightness_stream):
                brightness_out = F.conv2d(brightness_input, brightness_weight, stride=2, padding=3)
            
            color_stream.synchronize()
            brightness_stream.synchronize()
        
        torch.cuda.synchronize()
        parallel_time = time.perf_counter() - start
        print(f"CUDA Streams: {parallel_time*10:.2f}ms per call")
    else:
        print("\n=== CUDA Streams ===")
        print("CUDA Streams not supported on this device")
        parallel_time = sequential_time  # For comparison
    
    # Test 3: Grouped convolution
    print("\n=== Grouped Convolution ===")
    brightness_input_padded = F.pad(brightness_input, (0, 0, 0, 0, 0, 2))
    brightness_weight_padded = F.pad(brightness_weight, (0, 0, 0, 0, 0, 2))
    
    input_combined = torch.cat([color_input, brightness_input_padded], dim=1)
    weight_combined = torch.cat([color_weight, brightness_weight_padded], dim=0)
    
    start = time.perf_counter()
    for _ in range(100):
        out_combined = F.conv2d(input_combined, weight_combined, stride=2, padding=3, groups=2)
        color_out = out_combined[:, :64]
        brightness_out = out_combined[:, 64:]
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    grouped_time = time.perf_counter() - start
    print(f"Grouped Conv: {grouped_time*10:.2f}ms per call")
    
    # Results
    print(f"\n=== Results ===")
    print(f"Sequential: {sequential_time*10:.2f}ms")
    if supports_streams:
        print(f"CUDA Streams: {parallel_time*10:.2f}ms (speedup: {sequential_time/parallel_time:.2f}x)")
    print(f"Grouped Conv: {grouped_time*10:.2f}ms (speedup: {sequential_time/grouped_time:.2f}x)")


if __name__ == "__main__":
    test_sequential_vs_parallel_mac()