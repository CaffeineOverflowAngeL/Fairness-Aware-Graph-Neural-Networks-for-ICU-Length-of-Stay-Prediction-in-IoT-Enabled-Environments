import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths and corresponding time granularities
file_info = {
    "../evaluation_metrics_log_3days_12h_1h.csv": "1h",
    "../evaluation_metrics_log_3days_12h_2h.csv": "2h",
    "../evaluation_metrics_log_3days_12h_3h.csv": "3h",
}

# Load and merge data
df_list = []
for file_name, time_granularity in file_info.items():
    df = pd.read_csv(file_name)
    df["Time Granularity"] = time_granularity  # Add time granularity column
    df_list.append(df)

df_combined = pd.concat(df_list, ignore_index=True)

# IEEE-style blue color palette
ieee_palette = ["#0072B2", "#009E73", "#56B4E9", "#E69F00", "#F0E442", "#CC79A7"]
colors = sns.color_palette(ieee_palette, n_colors=len(df_combined["Model Name"].unique()))
color_map = {model: colors[i] for i, model in enumerate(df_combined["Model Name"].unique())}

# Set figure aesthetics
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "grid.alpha": 0.5,
})

# Create figure with two vertically stacked subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=False)

# Metrics to plot
metrics = ["AUC", "APR"]
titles = [r"AUC Across Sampling Granularities ($\delta t$)", 
          r"ARP Across Sampling Granularities ($\delta t$)"]

# Define line styles for fairness
line_styles = {True: "dotted", False: "solid"}

# Create legend elements manually
model_legend_elements = []
fairness_legend_elements = [
    plt.Line2D([0], [0], linestyle="solid", color="black", lw=2, label="Fairness = False"),
    plt.Line2D([0], [0], linestyle="dotted", color="black", lw=2, label="Fairness = True")
]

# Iterate over subplots
for ax, metric, title in zip(axes, metrics, titles):
    for model in df_combined["Model Name"].unique():
        model_color = color_map[model]  # Assign a unique IEEE color

        for fairness in df_combined["Fairness"].unique():
            subset = df_combined[(df_combined["Model Name"] == model) & (df_combined["Fairness"] == fairness)]
            ax.plot(
                subset["Time Granularity"],
                subset[metric],
                label=model if fairness is False else None,  # Only label the model once
                linestyle=line_styles[fairness],
                linewidth=2.5,
                marker="o",
                markersize=6,
                color=model_color
            )

        # Add only one entry per model to the legend
        if not any(leg.get_label() == model for leg in model_legend_elements):
            model_legend_elements.append(plt.Line2D([0], [0], color=model_color, lw=3, label=model))

    ax.set_ylabel(metric, fontsize=20)
    ax.set_title(title, fontsize=22, fontweight="bold", color="#00509E")  # IEEE Blue Title
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Adjust y-axis limits
    if metric == "AUC":
        ax.set_ylim(0.65, 0.90)  # Custom range for AUC
    else:
        ax.set_ylim(0.6, 0.85)   # Default for ARP

# Remove x-axis labels from the upper subplot
axes[0].label_outer()
axes[1].set_xlabel(r"Sampling Granularity ($\delta t$)", fontsize=20)  # Only on lower subplot

# Place model legend (colors) on the lower right of the upper subplot
axes[0].legend(handles=model_legend_elements, title="Models", loc="lower right", frameon=True, edgecolor="black", fontsize=16)

# Place fairness legend (line styles) on the upper right of the lower subplot
axes[1].legend(handles=fairness_legend_elements, title="Fairness Indicator", loc="upper right", frameon=True, edgecolor="black", fontsize=16)

# Adjust layout and show the plot
plt.tight_layout()
# Save the plot
output_path = "figure_4.png"
plt.savefig(output_path, dpi=500)
print(f"Plot saved to {output_path}")
