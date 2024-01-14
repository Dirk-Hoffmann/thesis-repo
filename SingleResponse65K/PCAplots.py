import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
from copy import copy
from joblib import load
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.io import loadmat
from utils.classes import SpectralDataset



data = loadmat('data/allMoistureDirk.mat')
print(data)

spectral_data = data['A']
wavelengths = data['nm'][0]     

# Create DataFrame for X
X = pd.DataFrame(spectral_data, columns=wavelengths)

print(X)

response_data = data['Y']

# Create DataFrame for Y
y = pd.DataFrame(response_data, columns=['Response'])  # Name the column as needed



#preprocess
XP = copy(X.to_numpy())
YP = copy(y.to_numpy())


#%% Mean center

XP_mean = np.mean(XP, axis=0)
YP_mean = np.mean(YP, axis=0)

XP-=XP_mean
YP-=YP_mean

# Calculate standard deviation for YP and YP_val
std_YP = np.std(YP, axis=0)

# Create filters based only on Y-values
filter_YP = np.all(abs(YP) < 3* std_YP, axis=1)

# Apply the filters to both X and Y datasets to ensure consistent lengths
XP = XP[filter_YP]
YP = YP[filter_YP]



# Run PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(XP)
explained_var = pca.explained_variance_ratio_ * 100  # As percentages

# Scatterplots
fig, axes = plt.subplots(1, 5, figsize=(30, 6))
plt.suptitle(f'Moisture Principal Component Analysis | n = {len(YP)} \n', fontsize = 16)

for i in range(0, 10, 2):
    ax = axes[i//2]
    ax.axline((0, 0), slope=1)
    ax.scatter(X_pca[:, i], X_pca[:, i+1], c=YP, cmap='viridis')
    ax.set_xlabel(f'PC{i+1} ({explained_var[i]:.2f}%)')
    ax.set_ylabel(f'PC{i+2} ({explained_var[i+1]:.2f}%)')
    ax.set_title(f'PC{i+1} vs PC{i+2}')
    ax.set_aspect('equal')
    ax.set_box_aspect(1)

plt.tight_layout()
plt.show()

plt.savefig(f'figures/moisture_pcaplots.png')
