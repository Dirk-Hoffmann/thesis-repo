import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the DataFrame
df = pd.read_csv('10runs_done.csv', index_col=0)

# Normalize the RMSEs by setting the lowest value in each category to 1
normalized_df = df.apply(lambda x: x / x.min(), axis=1)

# Initialize the figure with a larger size
plt.figure(figsize=(11, 13))  # Increased figure size

# Number of models
num_models = len(normalized_df.columns)

# Define a color palette, one color for each model
colors = plt.cm.Set3(np.linspace(0, 1, num_models))

# Iterate through the models to plot boxplots
for i, (model, color) in enumerate(zip(normalized_df.columns, colors)):
    # Data for the model
    model_data = [normalized_df.loc[ref, model] for ref in normalized_df.index.unique()]
    
    # Position of the boxplot
    positions = np.arange(len(normalized_df.index.unique())) + i * 0.15
    
    # Plotting horizontal boxplots with colors
    plt.boxplot(model_data, positions=positions, vert=False, widths=0.1, patch_artist=True,
                boxprops=dict(facecolor=color, color=color),
                flierprops=dict(markerfacecolor=color, marker='o', markersize=5, linestyle='none', markeredgecolor = 'none'),
                manage_ticks=False)

# Setting the y-axis ticks
plt.yticks(ticks=np.arange(len(normalized_df.index.unique())) + (num_models - 1) * 0.15 / 2, labels=normalized_df.index.unique())

plt.title('Normalized RMSE Distribution for Each Reference Across Models')
plt.xlabel('Normalized RMSE')
plt.ylabel('Reference')

# Set x-axis limit
plt.xlim(0.95, 6)

# Create custom legends and reverse the order
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
labels = list(normalized_df.columns)
plt.legend(handles[::-1], labels[::-1], loc='upper right')

plt.savefig('10runs_normalized_boxplot_horizontal_colored_outliers.png')
plt.show()
