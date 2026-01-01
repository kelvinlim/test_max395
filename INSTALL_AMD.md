# test_max395

Installation as of 20251231

Video my Hake Hardware

https://www.youtube.com/watch?v=vX7cYahvcPI

https://hakedev.substack.com/p/strix-halo-rocm-71-ubuntu-2404

https://gemini.google.com/u/1/app/ff21a52d739c0970  benchmarks windows vs. ubuntu

https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html

To set power high for apu
sudo rocm-smi --setperflevel high 

rocminfo

To view

https://github.com/Umio-Yasuno/amdgpu_top

amdgpu_top --gui



```
wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb
sudo apt install ./amdgpu-install_7.1.1.70101-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm

```

### AMD GPU Driver installation
```
wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb
sudo apt install ./amdgpu-install_7.1.1.70101-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
```

https://repo.radeon.com/rocm/

## venv

Software location directly from radeon - https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/

Uses this to get existing files

```
# the following commands pull the necessary packages directly from repo.radeon.com
# Direct install from AMD wheels (if the command above fails)
pip install --no-cache https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torch-2.9.1%2Brocm7.1.1.lw.git351ff442-cp312-cp312-linux_x86_64.whl \
            https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torchvision-0.24.0%2Brocm7.1.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
            https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torchaudio-2.9.0%2Brocm7.1.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
            https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/triton-3.5.1%2Brocm7.1.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
```
```
# this one loads torch 2.11.0
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1
# this installed torch 2.9.1 and removed torch 2.11.0
pip install torchvision torchaudio

```

```
# to get list of versions
source venv/bin/activate && pip index versions torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1 2>&1 | head -40

kolim@home-amd-max395:~/Projects/test_max395$ pip index versions torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1 
WARNING: pip index is currently an experimental command. It may be removed/changed in a future release without prior warning.
ERROR: No matching distribution found for torch
(.venv) kolim@home-amd-max395:~/Projects/test_max395$ 
```

## set performance to hi
```
sudo rocm-smi --setperflevel high
```