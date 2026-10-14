import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "ri5cy_unified_activation_results.csv"
OUTPUT_PLOT = "ri5cy_unified_activation_comparison.png"
Q_SCALE = 1024.0
PLOT_MIN = -5.0
PLOT_MAX = 5.0


def ideal_y(mode, x):
    if mode == 0:  # Swish
        return x / (1.0 + np.exp(-x))
    if mode == 1:  # GELU tanh approximation
        return 0.5 * x * (
            1.0
            + np.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
            )
        )
    if mode == 2:  # ReLU
        return np.maximum(0.0, x)
    if mode == 3:  # TanhExp, matching the supplied reference
        return x * np.tanh(np.exp(np.clip(x, -10.0, 10.0)))
    raise ValueError(f"Unsupported mode {mode}")


rows = {mode: [] for mode in range(4)}

with open(CSV_FILE, newline="") as csv_handle:
    reader = csv.DictReader(csv_handle)
    for row in reader:
        mode = int(row["MODE"])
        x_signed = int(row["X_SIGNED"])
        y_signed = int(row["Y_SIGNED"])
        rows[mode].append((x_signed / Q_SCALE, y_signed / Q_SCALE))

mode_names = {
    0: "Swish (MODE 00)",
    1: "GELU (MODE 01)",
    2: "ReLU (MODE 10)",
    3: "TanhExp (MODE 11)",
}

mse_scores = {}
prepared = {}

print("=" * 64)
print(f"MSE report over [{PLOT_MIN}, {PLOT_MAX}]")
print("=" * 64)

for mode in range(4):
    points = np.asarray(rows[mode], dtype=float)
    if points.size == 0:
        raise RuntimeError(f"No CSV samples found for MODE={mode}")

    order = np.argsort(points[:, 0])
    x_hw = points[order, 0]
    y_hw = points[order, 1]

    keep = (x_hw >= PLOT_MIN) & (x_hw <= PLOT_MAX)
    x_domain = x_hw[keep]
    y_domain = y_hw[keep]
    y_target = ideal_y(mode, x_domain)

    mse = float(np.mean((y_target - y_domain) ** 2))
    mse_scores[mode] = mse
    prepared[mode] = (x_domain, y_domain)
    print(f"{mode_names[mode]:<24} MSE: {mse:.8e}")

average_mse = float(np.mean(list(mse_scores.values())))
print("-" * 64)
print(f"Overall average MSE: {average_mse:.8e}")
print("=" * 64)

x_smooth = np.linspace(PLOT_MIN, PLOT_MAX, 2000)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for mode, axis in enumerate(axes.flat):
    x_hw, y_hw = prepared[mode]
    axis.plot(
        x_smooth,
        ideal_y(mode, x_smooth),
        "k--",
        linewidth=2.0,
        label="Ideal math",
    )
    axis.plot(
        x_hw,
        y_hw,
        "r-",
        linewidth=1.4,
        alpha=0.85,
        label=f"RI5CY full path\nMSE={mse_scores[mode]:.2e}",
    )
    axis.set_title(mode_names[mode], fontweight="bold")
    axis.set_xlabel("X_IN")
    axis.set_ylabel("Y_OUT")
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(loc="upper left")

fig.suptitle(
    "RI5CY Unified Activation Full-Path Verification\n"
    f"Overall average MSE: {average_mse:.2e}",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUTPUT_PLOT, dpi=300)
print(f"Saved plot: {OUTPUT_PLOT}")

# Export the compact format used by the supplied PyTorch flow.
database = {}
for mode in range(4):
    for x_float, y_float in rows[mode]:
        x_signed = int(round(x_float * Q_SCALE))
        y_signed = int(round(y_float * Q_SCALE))
        database.setdefault(x_signed, {})[mode] = y_signed

with open("cgp_results2.csv", "w", newline="") as output_handle:
    writer = csv.writer(output_handle)
    writer.writerow(["x_in", "swish_out", "gelu_out", "tanh_out", "relu_out"])
    for x_signed in sorted(database):
        outputs = database[x_signed]
        if all(mode in outputs for mode in range(4)):
            writer.writerow(
                [
                    x_signed,
                    outputs[0],
                    outputs[1],
                    outputs[3],
                    outputs[2],
                ]
            )

print("Saved PyTorch-compatible CSV: cgp_results2.csv")
