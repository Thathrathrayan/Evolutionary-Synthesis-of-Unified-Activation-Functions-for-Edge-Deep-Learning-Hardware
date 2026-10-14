import pandas as pd
import matplotlib.pyplot as plt

# Use Times New Roman throughout the plot
plt.rcParams["font.family"] = "Times New Roman"

# Avoid Type 3 fonts in PDF and EPS output
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

plt.rcParams["font.size"] = 16

df = pd.read_csv("resnet_tinyimagenet.csv")

# CSV column name : Legend name displayed in graph
legend_names = {
    "log_resnet18_CGPGELU": "CGPGELU (Proposed)",
    "log_resnet18_CGPSwish": "CGPSwish (Proposed)",
    "log_resnet18_CGPTanhexp": "CGPTanhexp (Proposed)",
    "log_resnet18_PolyGELU": "GELU (SOTA)",
    "log_resnet18_PolySwish": "Swish (SOTA)",
    "log_resnet18_PolyTanhexp": "Tanhexp (SOTA)"
}

plt.figure(figsize=(8, 5))

for col in df.columns[1:]:
    # Use the custom name if provided.
    # Otherwise, use the original column name.
    display_name = legend_names.get(col, col)

    plt.plot(
        df["epoch"],
        df[col],
        label=display_name,
        linewidth=1.5
    )

plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Training Loss", fontsize=16)
plt.title("Training Loss vs. Epoch", fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(
    title="Activation Function",
    loc="best",
    fontsize=14,
    title_fontsize=15
)

plt.grid(True)
plt.tight_layout()

# Save before plt.show()
plt.savefig(
    "resnet_tinyimagenet_loss.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "resnet_tinyimagenet_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
