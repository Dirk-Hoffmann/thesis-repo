import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from joblib import load
import pandas as pd
from Big8_testing.outdated_scripts.visualize_old import main
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.binning import bin_spectra_with_id
from utils.preprocess import detrend_parallel, snv_parallel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.binning import bin_spectra_with_id



class SpectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.to_numpy().astype(np.float32)
        self.y = y.to_numpy().astype(np.float32)
        #self.mask = ~np.isnan(self.y).astype(bool)
        self.wl = np.array(X.columns).astype(np.float32)
        self.response_names = y.columns
        # Perform preprocessing on entire dataset here
        self.X = self.preprocess(self.X)

    def preprocess(self, X):
        # Cropping
        X = X[:, np.where(self.wl >= 1100)[0]]
        
        # SNV, detrending, mean centering etc.
        snv_parallel(X)
        detrend_parallel(X)
        X = X - np.mean(X, axis=1, keepdims=True)
        # Z-score normalization
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = bin_spectra_with_id(X,bin_width=14)
        return X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y_row = self.y[idx]
        y_values = y_row[0:]
        #mask_value = self.mask[idx]
        
        return x, y_values

class MultiOutputCNN(nn.Module):
    def __init__(self, N, batch_size):
        super(MultiOutputCNN, self).__init__()
        self.conv1 = nn.Conv1d(batch_size, 32, 3)
        self.conv2 = nn.Conv1d(32, batch_size, 3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(int(((N - 2*3)/2)), 256)  # Assumes two Conv1D and one MaxPool1D layer
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 14)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        x = nn.ReLU()(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Concentrations can't be negative
        return x

def weights_init_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def weights_init_he(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

X = load('/home/djh/Big8_testing/data/big8/X.joblib')
y = load('/home/djh/Big8_testing/data/big8/Y.joblib')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)  


# Create dataset objects
train_dataset = SpectralDataset(X_train, y_train)
val_dataset = SpectralDataset(X_val, y_val)

print(train_dataset.__getitem__(0))

# Create DataLoader objects
batch_size = 512  # Choose an appropriate batch size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# Get the first batch of data to calculate N
first_data, y_values = next(iter(train_loader))
N = first_data.shape[-1] # Assuming the shape is [batch_size, N]


# Then instantiate the model
# Choose either xavier or he initialization
initialization = 'he'  # Change to 'he' for He initialization

#torch.autograd.set_detect_anomaly(True)

model = MultiOutputCNN(N, batch_size)


if initialization == 'xavier':
    model.apply(weights_init_xavier)
elif initialization == 'he':
    model.apply(weights_init_he)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss(reduction='none')
n_epochs = 300
train_losses = np.empty((1,N))
val_losses =[]
min_val_loss = float('inf')
patience = n_epochs/10
wait = 0
early_stopping = False

if __name__ == '__main__':
    wandb.init(
        project='CNN',
        config={
            'learning_rate':0.01,
            'architecture': 'CNN',
            'epochs':n_epochs,
        }
    )

    train_losses = []  # Initialize an empty list to hold epoch losses

    for epoch in range(n_epochs):
        print(f"{epoch} / {n_epochs}")
        batch_losses = []  # Initialize an empty list to hold batch losses for each epoch

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            target = torch.nan_to_num(target, nan=-1.0)

            mask = (target >= 0).float()
            loss = criterion(output, target)
            masked_loss = (loss * mask).sum() / mask.sum()

            batch_losses.append(masked_loss.item())  # Appending the detached masked loss to batch_losses list
            masked_loss.backward()
            optimizer.step()

            
        train_losses.append(batch_losses)  # Appending the batch_losses list to train_losses list
        
        with torch.no_grad():  # Disable gradient computation
            val_batch_losses = []
            for val_batch_idx, (val_data, val_target) in enumerate(val_loader):
                
                val_output = model(val_data)
                
                val_target = torch.nan_to_num(val_target, nan=-1.0)
                
                val_mask = (val_target >= 0).float()
                val_loss = criterion(val_output, val_target)
                
                val_masked_loss = (val_loss * val_mask).sum() / val_mask.sum()
                val_batch_losses.append(val_masked_loss.item())
                
            val_epoch_loss = np.mean(val_batch_losses)
            val_losses.append(val_epoch_loss)
            print(f"Validation Loss: {val_epoch_loss}")
      
        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
            wait = 0
            # Save the best model state
            torch.save(model.state_dict(), f'model_state_dict_{n_epochs}e.pth')
        elif early_stopping:
            wait += 1
            if wait >= patience:
                print("Early stopping due to no improvement.")
                break
        wandb.log({'batch_loss': np.mean(batch_losses),'val_loss':val_epoch_loss})
    mean_train_losses = [np.mean(x) for x in train_losses]
    mean_val_losses = val_losses
    std_losses = [np.std(x) for x in train_losses]
    epochs = range(len(mean_train_losses))

    plt.plot(epochs, mean_train_losses, label='Train Loss')
    plt.plot(epochs, mean_val_losses, label='Validation Loss')
    if early_stopping:
        plt.vlines(x=len(mean_train_losses)-patience, ymin=0, ymax=max(mean_train_losses), colors='red', label='Best Model')
    plt.title('Mean Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.ylim(bottom=0)

    plt.savefig(f'figures/CNN_train_val_losses.png')