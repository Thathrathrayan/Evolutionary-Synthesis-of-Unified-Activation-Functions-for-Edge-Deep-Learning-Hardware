Evolutionary Synthesis of Unified Activation Functions

This repository contains the Cartesian Genetic Programming (CGP) engine, generated RTL, functional verification scripts, CNN evaluation code, and RI5CY processor integration files for a unified hardware approximation of the Swish, GELU, and Tanhexp activation functions.

All implementation files are located in the codes/ directory.

Repository Structure
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

1. CGP Circuit Generation
Source File

codes/cgp_engine.cu

The CUDA-accelerated CGP engine evolves a unified logic circuit for the Swish, GELU, and Tanhexp activation functions. The selected CGP solution is automatically exported as synthesizable Verilog RTL.

Requirements
CUDA-compatible NVIDIA GPU
NVIDIA CUDA Toolkit with nvcc
Compile and Run

From the repository root:

nvcc -O3 codes/cgp_engine.cu -o codes/cgp_engine
./codes/cgp_engine


Alternatively, run the commands from the codes/ directory:

cd codes
nvcc -O3 cgp_engine.cu -o cgp_engine
./cgp_engine

Generated Output

The generated RTL is written to:

codes/unified_cuda_cgp_activation.v


The CGP engine periodically saves its evolutionary state. If a valid checkpoint is available, execution resumes from the saved generation.

2. CGP-Generated RTL Verification
Verilog RTL

codes/unified_cuda_cgp_activation.v

This synthesizable Verilog module implements the evolved activation-function circuit. It accepts a 16-bit fixed-point activation input and a function-selection input.

Verilog Testbench

codes/tb_universal.v

The testbench simulates the generated activation circuit.

Run the simulation from the codes/ directory:

cd codes
iverilog -o sim_run.vvp tb_universal.v unified_cuda_cgp_activation.v
vvp sim_run.vvp

Python Functional Verification

codes/cgp_act_fns_universal_tb.py

This script compares the generated RTL outputs against the expected quantized activation values.

Run it from the repository root:

python3 codes/cgp_act_fns_universal_tb.py

Verification Results

The corresponding results are stored in:

codes/cgp_results2.csv
codes/sim_results.csv
3. CNN Evaluation

The CNN experiments use quantization-aware training to compare polynomial-based and CGP-generated activation approximations.

Before running the experiments, configure the dataset paths and training parameters in the corresponding Python and shell scripts.

3.1. CIFAR-100

Relevant files:

codes/train_cifar100_resnet.py
codes/run_cifar100_resnet.sh

Run the CIFAR-100 activation experiments from the repository root:

chmod +x codes/run_cifar100_resnet.sh
./codes/run_cifar100_resnet.sh


Training logs are stored in:

codes/logs_cifar100/


The combined training-loss data is stored in:

codes/resnet_cifar100.csv

3.2. Tiny-ImageNet-200

Relevant files:

codes/train_tiny_imagenet.py
codes/run_tiny_imagenet_sweep.sh

Run the Tiny-ImageNet-200 activation experiments from the repository root:

chmod +x codes/run_tiny_imagenet_sweep.sh
./codes/run_tiny_imagenet_sweep.sh


Training logs are stored in:

codes/logs/


The combined training-loss data is stored in:

codes/resnet_tinyimagenet.csv

4. Loss Extraction and Plotting

The plotting scripts process the combined CNN training-loss data and generate loss curves for the evaluated activation functions.

4.1. CIFAR-100

Script: codes/combined_csv_plotter_cifar100.py

Input CSV: codes/resnet_cifar100.csv

Run:

python3 codes/combined_csv_plotter_cifar100.py

4.2. Tiny-ImageNet-200

Script: codes/combined_csv_plotter_tinyimagenet.py

Input CSV: codes/resnet_tinyimagenet.csv

Run:

python3 codes/combined_csv_plotter_tinyimagenet.py

5. RI5CY Integration and Verification

The repository includes a modified RI5CY processor implementation with integrated unified activation-function hardware and a custom instruction interface.

5.1. Software Requirements

The RI5CY simulation flow requires Verilator 5.008. Other versions may not compile or execute the provided simulation flow correctly.

Check the installed version:

verilator --version


Expected version:

Verilator 5.008

5.2. Integrated RTL

File: codes/ri5cy_uni_act_fns_fixed.sv

This SystemVerilog file contains the RI5CY RTL with the integrated unified activation-function hardware and custom instruction interface.

5.3. Activation Sweep Testbench

File: codes/ri5cy_unified_activation_sweep_tb.sv

This testbench evaluates the supported activation functions over a range of input values through the modified RI5CY core.

5.4. Run the Simulation

Script: codes/run_ri5cy_activation_sweep.sh

Run the complete compilation and activation-function simulation flow from the repository root:

chmod +x codes/run_ri5cy_activation_sweep.sh
./codes/run_ri5cy_activation_sweep.sh


The script invokes Verilator, builds the simulation model, runs the activation sweep, and writes the results to:

codes/ri5cy_unified_activation_results.csv


Verilator-generated build files are stored in:

codes/obj_dir/

5.5. Analyze the Results

Script: codes/analyze_ri5cy_activation_csv.py

Run the analysis script:

python3 codes/analyze_ri5cy_activation_csv.py


This script analyzes the activation-sweep results recorded in the generated CSV file.

6. Software Requirements

The complete workflow requires the following software and tools:

CUDA-compatible NVIDIA GPU
NVIDIA CUDA Toolkit (nvcc)
Python 3
PyTorch
TorchVision
Pandas
Matplotlib
Icarus Verilog
Verilator 5.008 for the RI5CY simulation flow
6.1. Install Python Dependencies
python3 -m pip install torch torchvision pandas matplotlib

6.2. Install Icarus Verilog

On Ubuntu, install Icarus Verilog using:

sudo apt update
sudo apt install iverilog

6.3. Install Verilator

Install Verilator separately and ensure that version 5.008 is available for the RI5CY simulation flow.

Verify the installation:

verilator --version

7. Typical Execution Workflow

The recommended workflow is:

Generate the circuit: Compile and run the CUDA-based CGP engine.
Verify the RTL: Simulate the generated Verilog and run the Python functional-verification script.
Evaluate CNN performance: Run the CIFAR-100 and Tiny-ImageNet-200 activation experiments.
Plot training losses: Generate the loss curves using the corresponding plotting scripts.
Verify RI5CY integration: Run the activation sweep using Verilator 5.008.
Analyze processor results: Process the generated RI5CY activation-sweep CSV file.
8. Code Availability

The repository provides the CGP engine, generated RTL, functional-verification scripts, CNN evaluation code, plotting utilities, and RI5CY processor-integration files used in this work.

All implementation files are available in the codes/ directory.
