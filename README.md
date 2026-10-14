# Evolutionary Synthesis of Unified Activation Functions for Deep-Learning Hardware

This repository contains the CGP implementation, generated RTL design,
functional-verification scripts, CNN evaluation code, and RI5CY
processor-integration files associated with the proposed unified
activation-function hardware.

Cartesian Genetic Programming (CGP) is used to generate a shared
logic-based implementation of Swish, GELU, and Tanhexp. The generated
design is evaluated through RTL simulation, quantization-aware CNN
training, and integration with the RI5CY RISC-V core.

## Repository Structure

```text
.
├── README.md
└── codes/
    ├── cgp_engine.cu
    ├── unified_cuda_cgp_activation.v
    ├── tb_universal.v
    ├── cgp_act_fns_universal_tb.py
    ├── cgp_results2.csv
    ├── sim_results.csv
    ├── train_cifar100_resnet.py
    ├── run_cifar100_resnet.sh
    ├── train_tiny_imagenet.py
    ├── run_tiny_imagenet_sweep.sh
    ├── combined_csv_plotter_cifar100.py
    ├── combined_csv_plotter_tinyimagenet.py
    ├── resnet_cifar100.csv
    ├── resnet_tinyimagenet.csv
    ├── logs/
    ├── logs_cifar100/
    ├── ri5cy_uni_act_fns_fixed.sv
    ├── ri5cy_unified_activation_sweep_tb.sv
    ├── run_ri5cy_activation_sweep.sh
    ├── ri5cy_unified_activation_results.csv
    └── analyze_ri5cy_activation_csv.py
```

## CGP Circuit Generation

The CUDA-based CGP implementation is provided in:

```text
codes/cgp_engine.cu
```

The program evolves a unified logic circuit for Swish, GELU, and
Tanhexp and exports the selected solution as synthesizable Verilog RTL.

Run the CGP engine using:

```bash
cd codes
nvcc -O3 cgp_engine.cu -o cgp_engine
./cgp_engine
```

The generated RTL is written to:

```text
unified_cuda_cgp_activation.v
```

## RTL Functional Verification

The generated unified activation circuit is provided in:

```text
codes/unified_cuda_cgp_activation.v
```

The associated Verilog and Python verification files are:

```text
codes/tb_universal.v
codes/cgp_act_fns_universal_tb.py
```

Run the Python-based functional verification using:

```bash
cd codes
python3 cgp_act_fns_universal_tb.py
```

The generated comparison results are stored in:

```text
codes/cgp_results2.csv
codes/sim_results.csv
```

## CNN Evaluation

The proposed and polynomial-based activation approximations are evaluated
using ResNet-18 with quantization-aware training on CIFAR-100 and
Tiny-ImageNet-200.

### CIFAR-100

The CIFAR-100 training files are:

```text
codes/train_cifar100_resnet.py
codes/run_cifar100_resnet.sh
```

Run all CIFAR-100 experiments using:

```bash
cd codes
chmod +x run_cifar100_resnet.sh
./run_cifar100_resnet.sh
```

The generated training logs are stored in:

```text
codes/logs_cifar100/
```

### Tiny-ImageNet-200

The Tiny-ImageNet-200 training files are:

```text
codes/train_tiny_imagenet.py
codes/run_tiny_imagenet_sweep.sh
```

Run all Tiny-ImageNet-200 experiments using:

```bash
cd codes
chmod +x run_tiny_imagenet_sweep.sh
./run_tiny_imagenet_sweep.sh
```

The generated training logs are stored in:

```text
codes/logs/
```

The dataset paths must be configured in the corresponding training
scripts before running the experiments.

## Training-Loss Plots

The combined CSV files containing the training-loss values are:

```text
codes/resnet_cifar100.csv
codes/resnet_tinyimagenet.csv
```

Generate the CIFAR-100 loss plot using:

```bash
cd codes
python3 combined_csv_plotter_cifar100.py
```

Generate the Tiny-ImageNet-200 loss plot using:

```bash
cd codes
python3 combined_csv_plotter_tinyimagenet.py
```

## RI5CY Processor Integration

The processor-level experiment integrates the unified activation hardware
with the RI5CY RISC-V core through a custom instruction interface.

The relevant files are:

```text
codes/ri5cy_uni_act_fns_fixed.sv
codes/ri5cy_unified_activation_sweep_tb.sv
codes/run_ri5cy_activation_sweep.sh
```

The RI5CY simulation flow requires:

```text
Verilator 5.008
```

Other Verilator versions may not compile or execute the provided flow
correctly.

Check the installed Verilator version using:

```bash
verilator --version
```

Run the complete RI5CY activation sweep using:

```bash
cd codes
chmod +x run_ri5cy_activation_sweep.sh
./run_ri5cy_activation_sweep.sh
```

The script automatically builds the RI5CY simulation, executes the
activation-function sweep, and stores the results in:

```text
codes/ri5cy_unified_activation_results.csv
```

Analyze the generated processor-level results using:

```bash
cd codes
python3 analyze_ri5cy_activation_csv.py
```

## Reproduction Flow

The complete evaluation can be reproduced in the following order:

1. Run the CUDA-based CGP engine.
2. Verify the generated Verilog RTL.
3. Run the CIFAR-100 and Tiny-ImageNet-200 CNN experiments.
4. Generate the training-loss plots.
5. Run the RI5CY activation sweep using Verilator 5.008.
6. Analyze the RI5CY simulation results.

## Requirements

The experiments require:

- CUDA-compatible NVIDIA GPU
- NVIDIA CUDA compiler
- Python 3
- PyTorch
- TorchVision
- Pandas
- Matplotlib
- Icarus Verilog
- Verilator 5.008
