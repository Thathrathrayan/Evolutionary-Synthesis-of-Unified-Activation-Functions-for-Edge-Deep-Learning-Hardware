#!/bin/bash

# You can adjust this batch size down to 64 if you get CUDA OOM errors
BATCH_SIZE=128 

activations=(
    "PolyTanhexp"
    "CGPTanhexp"
    "PolySwish"
    "CGPSwish"
    "PolyGELU"
    "CGPGELU"
)

mkdir -p logs_cifar100

for act in "${activations[@]}"; do
    echo "=================================================="
    echo " Starting ResNet-18 | Activation=$act"
    echo "=================================================="
    
    python3 train_cifar100_resnet.py --act "$act" --batch_size "$BATCH_SIZE"
    
    echo "Finished $act. Log saved to logs_cifar100/log_resnet18_${act}.txt"
    echo ""
done
