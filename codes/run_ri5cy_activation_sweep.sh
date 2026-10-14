#!/bin/sh
set -eu

STEP_VALUE="${STEP_VALUE:-64}"

rm -rf obj_dir

verilator --binary --timing -Wall -Wno-fatal \
  --top-module top_tb \
  -GSTEP="${STEP_VALUE}" \
  ri5cy_uni_act_fns_fixed.sv \
  ri5cy_unified_activation_sweep_tb.sv

./obj_dir/Vtop_tb
python3 analyze_ri5cy_activation_csv.py
