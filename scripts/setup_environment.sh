#!/bin/bash
# Environment Setup Script for DeepSeek R1 Reproduction
# Run this inside the ROCm Docker container

set -e

echo "=========================================="
echo "Setting up Open-R1 Environment"
echo "=========================================="

# Change to project directory
cd /home/asrr/amd_test

# Check if we're in a ROCm environment
if command -v rocm-smi &> /dev/null; then
    echo "ROCm detected!"
    rocm-smi --showproductname
else
    echo "WARNING: ROCm not detected. Make sure you're in the Docker container."
fi

# Install open-r1 in editable mode
echo ""
echo "Installing open-r1..."
cd /home/asrr/amd_test/open-r1
pip install -e ".[dev]" --quiet

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install --quiet \
    deepspeed==0.16.8 \
    accelerate==1.4.0 \
    wandb>=0.19.1 \
    liger-kernel>=0.5.10

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x /home/asrr/amd_test/scripts/*.sh

echo ""
echo "=========================================="
echo "Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. For SFT training:  ./scripts/train_sft.sh"
echo "  2. For GRPO training:"
echo "     - Terminal 1: ./scripts/start_vllm.sh"
echo "     - Terminal 2: ./scripts/train_grpo.sh"
echo ""
echo "Make sure to set up Weights & Biases:"
echo "  wandb login"
echo ""
