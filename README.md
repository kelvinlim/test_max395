# GPU Acceleration Test for AMD MAX 395+ with PyTorch

## Prerequisites

For AMD MAX 395+ GPU acceleration on Ubuntu 24.04, you need:

1. **ROCm (AMD's GPU compute platform)** - installed and configured
2. **PyTorch with ROCm support** - compiled for your specific GPU

## Installation Steps

### 1. Install ROCm Runtime

```bash
sudo apt update
sudo apt install rocm-core
```

### 2. Install PyTorch with ROCm Support

For AMD MAX 395+ (RDNA architecture), use:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

Or for the latest:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

### 3. Set Environment Variables (if needed)

```bash
export HSA_OVERRIDE_GFX_VERSION=gfx942
```

## Running the Test

```bash
python gpu_test.py
```

## Expected Output

When GPU acceleration is working, you should see:
- ✓ Device detection showing your AMD MAX 395+
- ✓ GPU tensor creation successful
- ✓ GPU performance significantly faster than CPU (2-10x speedup typical)
- ✓ GPU memory information displayed

## Troubleshooting

### "CUDA not available" message
- Ensure ROCm is installed: `rocm-smi --showid`
- Check if AMD MAX is detected: `rocm-smi`

### PyTorch not using GPU
- Verify PyTorch was installed with ROCm: `python -c "import torch; print(torch.cuda.is_available())"`
- Check which GPU device: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

### ROCm not detecting AMD MAX 395+
- Check driver: `dpkg -l | grep rocm`
- Verify GPU: `lspci | grep AMD`
- Check system logs: `dmesg | grep -i amd`

## Additional Resources

- [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/hip/overview.html)
- [AMD ROCm Installation Guide](https://rocmdocs.amd.com/)
- [AMD MAX 395 Specs](https://www.amd.com/en/products/specifications/processors/datacenter/amd-max-395.html)
