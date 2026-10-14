#!/bin/bash

#models=("resnet18" "mobilenetv2")
models=("resnet18")
activations=( 
    "PolyTanhexp" 
    "PolySwish"
    "PolyGELU"
    "CGPTanhexp"
    "CGPSwish"
    "CGPGELU"
)

mkdir -p logs

for model in "${models[@]}"; do
    for act in "${activations[@]}"; do
        echo "=================================================="
        echo " Starting: Model=$model | Activation=$act"
        echo "=================================================="
        
        python3 train_tiny_imagenet.py --model "$model" --act "$act"
        
        echo "Finished $model with $act. Log saved to logs/log_${model}_${act}.txt"
        echo ""
    done
done

