#!/bin/bash
# GPU Test Runner with various ROCm configuration options

cd /home/kolim/Projects/test_max395
source venv/bin/activate

echo "Running PyTorch GPU Test..."
echo "System ROCm Version: 7.1.1"
echo "PyTorch ROCm Version: 6.1"
echo ""

# Try different environment configurations
configs=(
    ""
    "HSA_OVERRIDE_GFX_VERSION=gfx1151"
    "TORCH_USE_HIP_DSA=1"
    "AMD_SERIALIZE_KERNEL=3"
)

for config in "${configs[@]}"; do
    echo "=========================================="
    if [ -z "$config" ]; then
        echo "Running with default settings..."
        python gpu_test.py
    else
        echo "Running with: $config"
        env "$config" python gpu_test.py
    fi
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Success!"
        exit 0
    fi
    echo ""
done

echo "All configurations failed. GPU may need additional setup."
exit 1
