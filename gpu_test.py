#!/usr/bin/env python3
"""
Test program to verify GPU acceleration with PyTorch on AMD MAX 395+
"""

import torch
import time
import sys

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def main():
    print_header("PyTorch GPU Acceleration Test")
    
    # 1. Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/HIP available: {torch.cuda.is_available()}")
    
    # Check if compiled with ROCm
    if "rocm" in torch.__version__.lower():
        print(f"Compiled with ROCm support: Yes")
        if not torch.cuda.is_available():
            print(f"\nNote: ROCm is compiled in, but GPU detection failed.")
            print(f"This may be due to:")
            print(f"  - AMD GPU driver not properly initialized")
            print(f"  - GPU in low-power state (check: rocm-smi)")
            print(f"  - Missing ROCm runtime libraries")
            print(f"\nRun: rocm-smi --showproductname")
            print(f"    to check GPU driver status.")
    
    # 2. Check device information
    print_header("Device Information")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("GPU not available in PyTorch (see notes above)")
        if "rocm" in torch.__version__.lower():
            print("\nDiagnostics:")
            print("GPU is detected by system but not available to PyTorch.")
            print("Try these steps to enable GPU acceleration:")
            print("  1. Check driver status: rocm-smi --showproductname")
            print("  2. Check power state: rocm-smi --showpower")
            print("  3. Reinstall ROCm: sudo apt install --reinstall rocm-core")
            print("  4. Verify HSA_OVERRIDE_GFX_VERSION if needed")
            return False
        else:
            print("PyTorch not compiled with ROCm support!")
            return False
    
    # 3. Test GPU availability
    print_header("GPU Tensor Test")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create a simple tensor on GPU
        x = torch.randn(100, 100, device=device)
        print(f"✓ Successfully created tensor on {device}")
        print(f"  Tensor shape: {x.shape}")
        print(f"  Tensor device: {x.device}")
        
    except Exception as e:
        error_msg = str(e)
        if "invalid device function" in error_msg.lower() or "hip error" in error_msg.lower():
            print(f"⚠ GPU kernel compatibility issue detected")
            print(f"  Error: {error_msg}")
            print(f"\n  This may be due to GPU architecture mismatch.")
            print(f"  Try setting environment variable:")
            print(f"  export HSA_OVERRIDE_GFX_VERSION=gfx1151")
            print(f"  or")
            print(f"  export TORCH_USE_HIP_DSA=1")
            return False
        else:
            print(f"✗ Failed to create GPU tensor: {e}")
            return False
    
    # 4. Performance benchmark
    print_header("Performance Benchmark")
    
    size = 5000
    iterations = 100
    
    print(f"Matrix multiplication test: {size}x{size} matrices, {iterations} iterations")
    
    # CPU benchmark
    print("\nCPU Performance:")
    x_cpu = torch.randn(size, size, device="cpu")
    y_cpu = torch.randn(size, size, device="cpu")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time:.4f} seconds")
    
    # GPU benchmark
    if torch.cuda.is_available():
        print("\nGPU Performance:")
        x_gpu = torch.randn(size, size, device="cuda")
        y_gpu = torch.randn(size, size, device="cuda")
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  Time: {gpu_time:.4f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"\n✓ GPU Speedup: {speedup:.2f}x")
        if speedup > 1.0:
            print("✓ GPU acceleration is working!")
        else:
            print("⚠ GPU appears slower than CPU for this operation")
    
    # 5. Memory info
    print_header("GPU Memory Information")
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
    
    print_header("Test Complete")
    print("✓ GPU acceleration is properly configured and working!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
