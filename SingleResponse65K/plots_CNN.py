import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from joblib import load
import pandas as pd
from visualize import main
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utils.binning import bin_spectra_with_id
from utils.preprocess import detrend_parallel, snv_parallel
from utils.mycroft_utils import rmse, sep, bias, optlv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from CNN_big import MultiOutputCNN, val_loader, val_dataset, train_dataset, N_EPOCHS, N, BATCH_SIZE, LEARNING_RATE, cmap
from utils.mycroft_utils import rmse
from utils.classes import SpectralDataset


model = MultiOutputCNN()

model.load_state_dict(torch.load(f'modelStateDict_{N_EPOCHS}_deep.pth'))


# Set the model to evaluation mode
model.eval()

true_vals = []
pred_vals = []


# Forward pass on validation set
with torch.no_grad():
    for data, target in val_loader:
        #print(data, target, mask_value)
        data = data.unsqueeze(1) #adding channel dimension to data
        output = model(data)
        true_vals.append(target.numpy())
        pred_vals.append(output.numpy())

# Convert list of arrays to single NumPy arrays
true_vals = val_dataset.inverse_transform_y(np.concatenate(true_vals))
pred_vals = train_dataset.inverse_transform_y(np.concatenate(pred_vals))

#Plotting

plt.figure(figsize=(7, 7))

plt.suptitle(f'Deep CNN Validation | Moisture', fontsize=16)


for i in range(true_vals.shape[1]):
    # Create mask for non-missing values
    mask = ~np.isnan(true_vals[:, i]) & ~np.isnan(pred_vals[:, i])
    
    # Exclude values outside of 3 standard deviations
    true_mean = np.nanmean(true_vals[:, i])
    true_std = np.nanstd(true_vals[:, i])
    pred_mean = np.nanmean(pred_vals[:, i])
    pred_std = np.nanstd(pred_vals[:, i])

    mask = mask & (true_vals[:, i] > true_mean - 3*true_std) & (true_vals[:, i] < true_mean + 3*true_std)
    mask = mask & (pred_vals[:, i] > pred_mean - 3*pred_std) & (pred_vals[:, i] < pred_mean + 3*pred_std)


    rounded_rmse = "{:.3f}".format(rmse(true_vals[mask,i], pred_vals[mask,i])[0], ndigits=3)

    print(rounded_rmse, N_EPOCHS)

    # print(f"Sample predictions for Output {i+1}: {pred_vals[mask][:10, i]}")
    # print(f"Range for true values (Output {i+1}): {np.min(true_vals[mask, i])} to {np.max(true_vals[mask, i])}")
    # print(f"Range for predictions (Output {i+1}): {np.min(pred_vals[mask, i])} to {np.max(pred_vals[mask, i])}")
    cmap_filtered = cmap[:8192][mask]

#    plt.subplot(4, 4, i+1)
    plt.scatter(true_vals[mask, i], pred_vals[mask, i], alpha=0.5, facecolors='none', edgecolors=cmap_filtered)

    # Identity line
    limits = [min(plt.xlim()[0], plt.ylim()[0]),
              max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(limits, limits, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.xlim(limits)
    plt.ylim(limits)

    plt.xlabel('True Values', fontsize = 14)
    plt.ylabel('Predictions', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize= 12)
    plt.title(f'RMSE = {rounded_rmse} | n = {len(pred_vals[mask, i])}',loc = 'left', fontsize = 14)

plt.tight_layout()
plt.savefig(f'figures/CNN_{N_EPOCHS}_epochs_{BATCH_SIZE}_BS.png')
plt.figure(figsize=(10, 10))

