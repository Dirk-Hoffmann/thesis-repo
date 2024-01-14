import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import copy
from joblib import load

# Load data
fileID = 'protein'
X_train, Y_train = load('/home/djh/Big8_testing/data/big8/X.joblib'), load('/home/djh/Big8_testing/data/big8/Y.joblib')
wl = np.array(X_train.columns).astype(np.float32)

label_font_size = 16
title_font_size = 18
tick_label_size = 14

# preprocess
XP = copy(X_train.iloc[:,np.where(wl >= 1100)[0]].to_numpy())

# Mean center X
XP_mean = np.mean(XP, axis=0)
XP -= XP_mean

# Run PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(XP)
explained_var = pca.explained_variance_ratio_ * 100  # As percentages

# Loop over each response in Y_train
for response in Y_train.columns:
    # Copy and preprocess Y
    YP = copy(Y_train[response].to_numpy())

    # Filter out missing values in YP
    missing_values_filter = ~np.isnan(YP)  # Create a filter for non-missing values
    YP = YP[missing_values_filter]
    X_pca_filtered = X_pca[missing_values_filter]  # Apply the same filter to X_pca

    # Check if YP is empty after filtering
    if len(YP) == 0:
        continue  # Skip this response if no data left after filtering

    # Continue with the rest of the processing
    YP_mean = np.mean(YP)
    YP -= YP_mean

    # Standard deviation and filter for Y
    std_YP = np.std(YP)
    filter_YP = abs(YP) < 5 * std_YP

    # Apply filter
    X_pca_filtered = X_pca_filtered[filter_YP]
    YP_filtered = YP[filter_YP]

    # Scatterplots
    fig, axes = plt.subplots(1, 5, figsize=(30, 7))
    fig.suptitle(f'PCA Plots for Response: {response} | n= {len(YP_filtered)}', fontsize=title_font_size)  # Set font size for main title

    for i in range(0, 10, 2):
        ax = axes[i//2]
        ax.axline((0, 0), slope=1)
        scatter = ax.scatter(X_pca_filtered[:, i], X_pca_filtered[:, i+1], c=YP_filtered, cmap='viridis')
        ax.set_xlabel(f'PC{i+1} ({explained_var[i]:.2f}%)', fontsize=label_font_size)
        ax.set_ylabel(f'PC{i+2} ({explained_var[i+1]:.2f}%)', fontsize=label_font_size)
        ax.set_title(f'PC{i+1} vs PC{i+2}', fontsize=title_font_size)
        ax.set_aspect('equal')
        ax.set_box_aspect(1)

        # Set font size for tick labels
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust layout for main title
    plt.savefig(f'figures/PCAPlots/{fileID}_{response}_pcaplots.png')
    plt.close(fig)  # Close the figure to free memory
