import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# # Read data from CSV
# df = pd.read_csv('shallowCNNLosses.csv')

# # List of columns to plot, excluding the first three columns
# columns_to_plot = df.columns[3:]

# # Set the style of the seaborn plot
# sns.set(style="ticks")

# # Number of rows and columns for the subplot grid
# n_rows = 4
# n_cols = 4

# # Create a figure and a grid of subplots
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

# plt.suptitle('RMSE for Sigmoid activation vs ReLU activation', fontsize=16, y=0.98)

# # Iterate over the rows and columns and create each plot
# for i in range(n_rows):
#     for j in range(n_cols):
#         # Calculate the index of the current column
#         idx = i * n_cols + j

#         # Check if the current index is within the range of columns to plot
#         if idx < len(columns_to_plot):
#             col = columns_to_plot[idx]
#             sns.lineplot(data=df, x='Hidden_layer_size', y=col, hue="Activation_function", ax=axes[i, j], linestyle=':', marker = 'o')
            
#             # Set x-axis to logarithmic scale
#             axes[i, j].set(xscale="log")
#             axes[i,j].legend(fontsize = 'xx-small')

#             # Perform ANOVA and get the F-statistic
#             f_val, p_val = stats.f_oneway(df[df['Activation_function'] == 'Sigmoid'][col],
#                                           df[df['Activation_function'] == 'ReLU'][col])
            
#             # Display the F-statistic and p-value
#             axes[i, j].set_title(f"{col} (F={f_val:.2f}, p={p_val:.3f})", fontsize='xx-small')

#         else:
#             axes[i, j].axis('off')  # Turn off axis if no more columns to plot

# # Adjust layout
# plt.tight_layout()

# # Save the plot
# plt.savefig('Lossplotting.png')

# # Show the plot
# plt.show()

from sklearn.preprocessing import MinMaxScaler

# Read data from CSV
df = pd.read_csv('shallowCNNLosses.csv')

# Filter the DataFrame to only include rows where Activation_function is 'ReLU'
relu_df = df[df['Activation_function'] == 'ReLU'].copy()
# Columns that represent RMSE values
rmse_columns = relu_df.columns[3:]  # Adjust the slice to match your RMSE columns

# Normalize the RMSE columns
scaler = MinMaxScaler()
relu_df[rmse_columns] = scaler.fit_transform(relu_df[rmse_columns])

# Set the style of the seaborn plot
sns.set(style="ticks")

# Create a line plot
plt.figure(figsize=(10, 6))

# Iterate over each RMSE category and plot
for col in rmse_columns:
    sns.lineplot(data=relu_df, x='Hidden_layer_size', y=col, label=col, marker='o')

plt.xlabel('Hidden Layer Size')
plt.ylabel('Normalized RMSE')
plt.title('Normalized RMSE for ReLU Activation Function')
plt.yscale('log')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.legend(title='RMSE Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('Normalized_ReLU_RMSE_Plot.png')

# Show the plot
plt.show()
