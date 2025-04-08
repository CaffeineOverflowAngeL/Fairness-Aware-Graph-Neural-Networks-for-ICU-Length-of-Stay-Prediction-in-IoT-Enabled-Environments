import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

# Load the CSV file
file_path = "/mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/evaluation_metrics_log.csv"
df = pd.read_csv(file_path)

# Extract relevant data
x = df["APR"]
y = df["AUC"]
sizes = df["Ops (M)"]  # Circle size based on Ops (M)
fairness = df["Fairness"]  # Determines border style
batch_sizes = df["Batch Size"]

print("Before: ", sizes)

# Normalize sizes for better visualization
sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 1000 + 100

print("After:", sizes)

# Create figure with a modern style
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor("#f5f5f5")

# Scale sizes properly
size_scale = 1500 / max(sizes)  # Scale factor for circle sizes

# Plot each point with the requested style
for i in range(len(df)):
    # Main circle
    plt.scatter(
        x[i],
        y[i],
        s=sizes[i] * size_scale,
        color="#3498db",  # Default blue color
        alpha=0.8,
        edgecolors="black",
        linewidth=1.5
    )

    # Fairness marker (Star for True, Inner Circle for False)
    if fairness[i]:  # Star marker for fairness True
        plt.scatter(x[i], y[i], s=50, color="white", marker="*", edgecolors="black", linewidth=1.5)
    else:  # Small inner circle for fairness False
        plt.scatter(x[i], y[i], s=50, color="black", alpha=0.7)

    # Move batch size label on top of the circle
    if batch_sizes[i] == 512:
        if fairness[i]:
            plt.text(
                x[i],
                y[i] + 0.002,  # Small offset upwards
                str(batch_sizes[i]),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black"
            )
        else:
            plt.text(
                x[i],
                y[i] - 0.00325,  # Small offset downwards
                str(batch_sizes[i]),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black"
            )
    elif batch_sizes[i] == 256:
        plt.text(
            x[i],
            y[i] + 0.0015,  # Small offset upwards
            str(batch_sizes[i]),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black"
        )
    elif batch_sizes[i] == 128:
        plt.text(
            x[i],
            y[i] + 0.00125,  # Small offset upwards
            str(batch_sizes[i]),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black"
        )
    elif batch_sizes[i] <= 128 and batch_sizes[i] > 16:
        plt.text(
            x[i],
            y[i] + 0.0008,  # Small offset upwards
            str(batch_sizes[i]),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black"
        )
    else:
        if fairness[i]:
            plt.text(
                x[i],
                y[i] + 0.0007,  # Small offset upwards
                str(batch_sizes[i]),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black"
            )
        else:
            plt.text(
                x[i],
                y[i] - 0.002,  # Small offset downwards
                str(batch_sizes[i]),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black"
            )


# Define legend placement range
legend_x_values = np.linspace(0.6925, 0.7175, 4)  # Four circles spread from 0.69 to 0.73
legend_y_value = 0.81  # Fixed y position

# Define Ops values for legend
legend_ops_values = [min(sizes), max(sizes) / 4, max(sizes) / 2, max(sizes)]
legend_color = "#7f8c8d"  # Faded dark gray

# Plot legend circles horizontally
for i, (x_pos, ops_val) in enumerate(zip(legend_x_values, legend_ops_values)):
    # Main circle for Ops scale
    plt.scatter(
        x_pos,
        legend_y_value,
        s=ops_val * size_scale,
        color=legend_color,
        alpha=0.5,
        edgecolors="black",
        linewidth=1.5
    )

    # Add markers **only for the middle two**
    if i == 1:  # 2nd circle (Fairness=False)
        plt.scatter(x_pos, legend_y_value, s=50, color="black", alpha=0.7)  # Inner circle
    elif i == 2:  # 3rd circle (Fairness=True)
        plt.scatter(x_pos, legend_y_value, s=50, color="white", marker="*", edgecolors="black", linewidth=1.5)  # Star

    print(ops_val)
    denormalized_ops_val = ((ops_val - 100) / 1000) * (df["Ops (M)"].max() - df["Ops (M)"].min()) + df["Ops (M)"].min()
    plt.text(x_pos, legend_y_value - 0.004, f"{int(denormalized_ops_val)}M", fontsize=16, ha="center", color="black")


# Add Fairness labels **above** the corresponding legend circles
plt.text(legend_x_values[1], legend_y_value + 0.002, "Unfair", fontsize=16, ha="center", color="black")
plt.text(legend_x_values[2], legend_y_value + 0.002, "Fair", fontsize=16, ha="center", color="black")

# Labels and title
ax.set_xlabel("ARP", fontsize=20, fontweight="bold", color="#333")
ax.set_ylabel("AUC", fontsize=20, fontweight="bold", color="#333")
ax.set_title("Batch Size Impact on ARP, AUC, Fairness and FLOPs(M)", fontsize=22, fontweight="bold", color="#222")

# Extend y-axis to 0.85 to prevent overlap
ax.set_ylim(min(y) - 0.01, 0.852)
ax.tick_params(axis='both', which='major', labelsize=18)

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save the plot
output_path = "figure_3.png"
plt.savefig(output_path, dpi=500)
print(f"Plot saved to {output_path}")
