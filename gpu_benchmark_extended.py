#!/usr/bin/env python3
"""
Extended GPU benchmark for AMD Strix Halo (Radeon MAX 395+)
Tests various workload sizes and ML-specific operations
"""

import torch
import time
import sys

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def benchmark_matmul(size, iterations, device_name):
    """Benchmark matrix multiplication"""
    device = torch.device(device_name)
    x = torch.randn(size, size, device=device, dtype=torch.float32)
    y = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        _ = torch.matmul(x, y)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        result = torch.matmul(x, y)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed

def benchmark_conv2d(batch_size, iterations, device_name):
    """Benchmark 2D convolution (common in CNNs)"""
    device = torch.device(device_name)
    
    # Create a typical CNN layer setup
    input_tensor = torch.randn(batch_size, 64, 224, 224, device=device)
    conv_layer = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
    
    # Warmup
    for _ in range(3):
        _ = conv_layer(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        output = conv_layer(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed

def benchmark_fp16_matmul(size, iterations, device_name):
    """Benchmark FP16 matrix multiplication (faster on modern GPUs)"""
    device = torch.device(device_name)
    
    if device.type == 'cpu':
        # FP16 not well supported on CPU
        return None
    
    x = torch.randn(size, size, device=device, dtype=torch.float16)
    y = torch.randn(size, size, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        _ = torch.matmul(x, y)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        result = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed

def benchmark_element_wise(size, iterations, device_name):
    """Benchmark element-wise operations"""
    device = torch.device(device_name)
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = torch.relu(x * y + x - y)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed

def run_matrix_size_tests():
    """Test different matrix sizes to find GPU sweet spot"""
    print_header("Matrix Multiplication - Size Scaling")
    
    sizes = [1000, 2000, 5000, 8000, 10000, 15000]
    iterations_map = {1000: 100, 2000: 100, 5000: 50, 8000: 20, 10000: 10, 15000: 5}
    
    print(f"{'Size':<10} {'Iterations':<12} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for size in sizes:
        iterations = iterations_map[size]
        
        try:
            print(f"{size}x{size:<5} {iterations:<12}", end=" ", flush=True)
            
            # CPU benchmark
            cpu_time = benchmark_matmul(size, iterations, "cpu")
            print(f"{cpu_time:>10.4f}s", end=" ", flush=True)
            
            # GPU benchmark
            if torch.cuda.is_available():
                gpu_time = benchmark_matmul(size, iterations, "cuda")
                speedup = cpu_time / gpu_time
                print(f"{gpu_time:>10.4f}s {speedup:>9.2f}x", flush=True)
            else:
                print("GPU N/A")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠ Out of memory")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"  ⚠ Error: {e}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")

def run_ml_workload_tests():
    """Test ML-specific workloads"""
    print_header("ML Workload Benchmarks")
    
    # Convolution test
    print("2D Convolution (CNN layer):")
    print(f"{'Batch Size':<12} {'Iterations':<12} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
    print("-" * 70)
    
    batch_sizes = [1, 4, 8, 16, 32]
    conv_iterations = 20
    
    for batch_size in batch_sizes:
        try:
            print(f"{batch_size:<12} {conv_iterations:<12}", end=" ", flush=True)
            
            cpu_time = benchmark_conv2d(batch_size, conv_iterations, "cpu")
            print(f"{cpu_time:>10.4f}s", end=" ", flush=True)
            
            if torch.cuda.is_available():
                gpu_time = benchmark_conv2d(batch_size, conv_iterations, "cuda")
                speedup = cpu_time / gpu_time
                print(f"{gpu_time:>10.4f}s {speedup:>9.2f}x")
            else:
                print("GPU N/A")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠ Out of memory")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"  ⚠ Error: {e}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")

def run_fp16_tests():
    """Test FP16 performance (should be faster on GPU)"""
    if not torch.cuda.is_available():
        return
    
    print_header("FP16 (Half Precision) Performance")
    
    sizes = [5000, 8000, 10000, 15000]
    iterations_map = {5000: 50, 8000: 30, 10000: 20, 15000: 10}
    
    print(f"{'Size':<10} {'Iterations':<12} {'FP32 Time':<12} {'FP16 Time':<12} {'FP16 Speedup':<15}")
    print("-" * 70)
    
    for size in sizes:
        iterations = iterations_map[size]
        
        try:
            print(f"{size}x{size:<5} {iterations:<12}", end=" ", flush=True)
            
            # FP32 benchmark
            fp32_time = benchmark_matmul(size, iterations, "cuda")
            print(f"{fp32_time:>10.4f}s", end=" ", flush=True)
            
            # FP16 benchmark
            fp16_time = benchmark_fp16_matmul(size, iterations, "cuda")
            if fp16_time:
                speedup = fp32_time / fp16_time
                print(f"{fp16_time:>10.4f}s {speedup:>13.2f}x")
            else:
                print("N/A")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠ Out of memory")
                torch.cuda.empty_cache()
            else:
                print(f"  ⚠ Error: {e}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")

def run_element_wise_tests():
    """Test element-wise operations (high memory bandwidth)"""
    print_header("Element-wise Operations")
    
    sizes = [5000, 10000, 15000, 20000]
    iterations = 1000
    
    print(f"{'Size':<10} {'Iterations':<12} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for size in sizes:
        try:
            print(f"{size}x{size:<5} {iterations:<12}", end=" ", flush=True)
            
            cpu_time = benchmark_element_wise(size, iterations, "cpu")
            print(f"{cpu_time:>10.4f}s", end=" ", flush=True)
            
            if torch.cuda.is_available():
                gpu_time = benchmark_element_wise(size, iterations, "cuda")
                speedup = cpu_time / gpu_time
                print(f"{gpu_time:>10.4f}s {speedup:>9.2f}x")
            else:
                print("GPU N/A")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠ Out of memory")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"  ⚠ Error: {e}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")

def main():
    print_header("Extended GPU Benchmark - AMD Strix Halo")
    
    # System info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/HIP available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    else:
        print("⚠ GPU not available - CPU-only benchmarks will run")
        print("\nTo enable GPU, try:")
        print("  sudo rocm-smi --setperflevel high")
        print("  export HSA_OVERRIDE_GFX_VERSION=gfx1151")
        return False
    
    # Run benchmark suites
    try:
        run_matrix_size_tests()
        run_ml_workload_tests()
        run_fp16_tests()
        run_element_wise_tests()
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        return False
    
    print_header("Benchmark Complete")
    print("💡 Observations for Integrated GPUs (like Strix Halo):")
    print("  • Larger workloads typically show better GPU speedup")
    print("  • Convolution operations often benefit more from GPU")
    print("  • FP16 operations may provide significant speedup")
    print("  • Small matrices may be faster on CPU due to overhead")
    print("\n💡 To improve GPU performance:")
    print("  • Ensure high performance mode: sudo rocm-smi --setperflevel high")
    print("  • Use larger batch sizes for ML training")
    print("  • Consider FP16/mixed precision training")
    print("  • Check ROCm driver optimization for your GPU arch")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
