import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION BLOCK (Change these for different .v files)
# =====================================================
VERILOG_FILE  = "unified_cuda_cgp_activation.v"
MODULE_NAME   = "unified_input_quantized_activation"
ENCODING_MODE = "TWOS_COMPLEMENT"  # Options: "SIGN_MAGNITUDE" or "TWOS_COMPLEMENT"
OUTPUT_PLOT   = "unified_input_quantized_activation.png"

# =====================================================
# 1. Generate Dynamic Verilog Testbench
# =====================================================
tb_code = f"""
`timescale 1ns / 1ps
module tb_universal;
    reg [15:0] X_IN;
    reg [1:0] MODE;
    wire [15:0] Y_OUT;
    
    integer f, i, m;

    {MODULE_NAME} dut (
        .X_IN(X_IN),
        .MODE(MODE),
        .Y_OUT(Y_OUT)
    );

    initial begin
        f = $fopen("sim_results.csv", "w");
        $fwrite(f, "MODE,X_IN,Y_OUT\\n");
        for (m = 0; m < 4; m = m + 1) begin
            MODE = m;
            for (i = 0; i < 65536; i = i + 64) begin
                X_IN = i;
                #1;
                $fwrite(f, "%0d,%0d,%0d\\n", MODE, X_IN, Y_OUT);
            end
        end
        $fclose(f);
        $finish;
    end
endmodule
"""

with open("tb_universal.v", "w") as f:
    f.write(tb_code)

print(f"[+] Generated tb_universal.v targeting module: {MODULE_NAME}")

# Compile & Execute with iverilog
vvp_out = "sim_run.vvp"
subprocess.run(["iverilog", "-o", vvp_out, VERILOG_FILE, "tb_universal.v"], check=True)
subprocess.run(["vvp", vvp_out], check=True)
print("[+] Simulation complete. Processing results...")

# =====================================================
# 2. Number Decoding (Handles Two's Comp & Sign-Mag)
# =====================================================
def decode_q510(val, encoding="TWOS_COMPLEMENT"):
    if encoding == "TWOS_COMPLEMENT":
        if val >= 32768:
            val -= 65536
        return val / 1024.0
    elif encoding == "SIGN_MAGNITUDE":
        sign_bit = (val >> 15) & 1
        magnitude = val & 0x7FFF
        float_val = magnitude / 1024.0
        return -float_val if sign_bit else float_val

# =====================================================
# 3. Ideal Target Math Functions
# =====================================================
def get_ideal_y(mode, x_array):
    if mode == 0:   # Swish
        return x_array * (1.0 / (1.0 + np.exp(-x_array)))
    elif mode == 1: # GELU
        return 0.5 * x_array * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x_array + 0.044715 * x_array**3)))
    elif mode == 2: # ReLU
        return np.maximum(0.0, x_array)
    elif mode == 3: # TanhExp
        return x_array * np.tanh(np.exp(np.clip(x_array, -10.0, 10.0)))

# =====================================================
# 4. Parse CSV & Calculate Mean Squared Error (MSE)
# =====================================================
hw_results = {0: [], 1: [], 2: [], 3: []}
x_vals = {0: [], 1: [], 2: [], 3: []}

with open("sim_results.csv", "r") as f:
    next(f) # Skip header
    for line in f:
        mode, x_in, y_out = map(int, line.strip().split(','))
        x_f = decode_q510(x_in, encoding="TWOS_COMPLEMENT")
        y_f = decode_q510(y_out, encoding=ENCODING_MODE)
        if -5.0 <= x_f <= 5.0:
            x_vals[mode].append(x_f)
            hw_results[mode].append(y_f)

mode_names = {0: "Swish (MODE 00)", 1: "GELU (MODE 01)", 2: "ReLU (MODE 10)", 3: "TanhExp (MODE 11)"}
mse_scores = {}

print("\n" + "="*50)
print("MEAN SQUARED ERROR (MSE) REPORT (Domain: [-5.0, +5.0])")
print("="*50)

for m in range(4):
    sort_idx = np.argsort(x_vals[m])
    x_hw = np.array(x_vals[m])[sort_idx]
    y_hw = np.array(hw_results[m])[sort_idx]
    y_target = get_ideal_y(m, x_hw)
    mse = np.mean((y_target - y_hw) ** 2)
    mse_scores[m] = mse
    print(f" -> {mode_names[m]:<20} MSE: {mse:.6e}")

avg_mse = np.mean(list(mse_scores.values()))
print("-" * 50)
print(f" OVERALL AVERAGE MSE: {avg_mse:.6e}")
print("="*50 + "\n")

# =====================================================
# 5. Plot Comparison Grid
# =====================================================
x_ideal_smooth = np.linspace(-5, 5, 1000)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for m in range(4):
    title = mode_names[m]
    y_ideal_smooth = get_ideal_y(m, x_ideal_smooth)
    sort_idx = np.argsort(x_vals[m])
    x_hw = np.array(x_vals[m])[sort_idx]
    y_hw = np.array(hw_results[m])[sort_idx]
    axs[m].plot(x_ideal_smooth, y_ideal_smooth, 'k--', label="Ideal Math", linewidth=2.0)
    axs[m].plot(x_hw, y_hw, 'r-', label=f"CGP Verilog\n(MSE: {mse_scores[m]:.2e})", alpha=0.85, linewidth=1.5)
    axs[m].set_title(title, fontweight='bold', fontsize=12)
    axs[m].set_xlabel("X_IN", fontsize=10)
    axs[m].set_ylabel("Y_OUT", fontsize=10)
    axs[m].grid(True, linestyle=':', alpha=0.6)
    axs[m].legend(loc="upper left")

plt.suptitle(f"Hardware Verification & Precision Profile: {MODULE_NAME}\nOverall Avg MSE: {avg_mse:.2e}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.show()
print(f"[+] Comparison graph successfully saved as '{OUTPUT_PLOT}'.")

# =====================================================
# 6. Export Compatible 'cgp_results2.csv' for PyTorch
# =====================================================
print("\n[+] Exporting PyTorch-compatible 'cgp_results2.csv'...")

db = {}
with open("sim_results.csv", "r") as f:
    next(f) # Skip header
    for line in f:
        mode, x_in, y_out = map(int, line.strip().split(','))
        
        # Convert 16-bit raw X_IN to standard Python signed integer
        x_signed = x_in - 65536 if x_in >= 32768 else x_in
        
        # Convert 16-bit raw Y_OUT to standard Python signed integer
        if ENCODING_MODE == "TWOS_COMPLEMENT":
            y_signed = y_out - 65536 if y_out >= 32768 else y_out
        elif ENCODING_MODE == "SIGN_MAGNITUDE":
            sign_bit = (y_out >> 15) & 1
            magnitude = y_out & 0x7FFF
            y_signed = -magnitude if sign_bit else magnitude
            
        if x_signed not in db:
            db[x_signed] = {0: 0, 1: 0, 2: 0, 3: 0}
        db[x_signed][mode] = y_signed

# Write exactly matching the HDC Verilog output format
with open("cgp_results2.csv", "w") as f:
    f.write("x_in,swish_out,gelu_out,tanh_out,relu_out\n")
    # Sort strictly by x_signed so PyTorch searchsorted doesn't crash
    for x_signed in sorted(db.keys()):
        y_swish = db[x_signed][0] # Mode 0
        y_gelu  = db[x_signed][1] # Mode 1
        y_tanh  = db[x_signed][3] # Mode 3 (TanhExp in CGP)
        y_relu  = db[x_signed][2] # Mode 2 (ReLU in CGP)
        f.write(f"{x_signed},{y_swish},{y_gelu},{y_tanh},{y_relu}\n")

print("[+] Successfully generated 'cgp_results2.csv'!")
