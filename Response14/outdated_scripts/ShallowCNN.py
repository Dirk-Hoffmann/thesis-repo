
# Hyperparameters
BATCH_SIZE = 128
N_EPOCHS = 8000
HIDDEN_LAYER_SIZE = 3*14
LEARNING_RATE = 0.001
TEST_SIZE = 0.4
RANDOM_STATE = 42
PATIENCE = N_EPOCHS // 10
EARLY_STOPPING = True


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
from copy import copy
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utils.binning import bin_spectra_with_id
from utils.preprocess import detrend_parallel, snv_parallel


class SpectralDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X.to_numpy().astype(np.float32), y.to_numpy().astype(np.float32)
        self.wl = np.array(X.columns).astype(np.float32)
        self.response_names = y.columns
        # Compute min and max for y for min-max scaling
        self.y_min = np.nanmin(self.y, axis = 0)
        self.y_max = np.nanmax(self.y, axis = 0)

        # Normalize y
        self.y = (self.y - self.y_min) / (self.y_max - self.y_min)

        self.X = self.preprocess(self.X)

    def preprocess(self, X):
        processed_X = X[:, np.where(self.wl >= 1100)[0]]
        snv_parallel(processed_X)
        detrend_parallel(processed_X)
        processed_X -= np.mean(processed_X, axis=1, keepdims=True)
        scaler = StandardScaler()
        processed_X = bin_spectra_with_id(scaler.fit_transform(processed_X), bin_width=10) #Optisk opløsning på apparatet er ca. 10 nm, en måling for hver .5 nm
        return processed_X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = self.X[idx], self.y[idx, 0:]
        return item
    
    def inverse_transform_y(self, y_normalized):
        # Denormalize y
        return y_normalized * (self.y_max - self.y_min) + self.y_min


class ShallowCNN(nn.Module):
    def __init__(self, N, channels= 1):
        super(ShallowCNN, self).__init__()

        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(channels, 128, 3, padding=1)
        
        # self.conv1_output_size = self.calculate_conv_output(N) 
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(140, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, 14)


    # def calculate_conv_output(self, N):
    #     # Apply Conv1d output size formula
    #     conv1_output = (N + 2 * 1 - 3) / 1 + 1  # Add padding, remove kernel size, stride is 1.
    #     # Apply MaxPool1d reduction
    #     pool_output = conv1_output / 2
    #     return int(pool_output)  # Cast to int, because the output size must be an integer.


    def forward(self, x):
        # x = nn.ReLU()(self.conv1(x))
        # x = self.pool(x)
        # x = x.view(x.size(0), -1)
        # x = nn.ReLU()(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        # return x
        print(f'input shape: {x.shape}')
        x = nn.ReLU()(self.conv1(x))
        print(f"Shape after conv1: {x.shape}")
        x = self.pool(x)
        print(f"Shape after pool: {x.shape}")
        x = x.view(x.size(0), -1)  # Make sure this is correct
        print(f"Shape after view: {x.shape}")
        x = nn.ReLU()(self.fc1(x))
        print(f"Shape after fc1: {x.shape}")
        x = self.fc2(x)
        print(f"Shape after fc2: {x.shape}")
        return x


def init_weights(m, method='he'):
    init_fn = nn.init.xavier_uniform_ if method == 'xavier' else nn.init.kaiming_uniform_
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init_fn(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

X, y = load('/home/djh/Big8_testing/data/big8/X.joblib'), load('/home/djh/Big8_testing/data/big8/Y.joblib')
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)
train_dataset = SpectralDataset(X_train, y_train)
test_dataset = SpectralDataset(X_test, y_test)
val_dataset = SpectralDataset(X_val, y_val)

print(train_dataset.response_names)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle= False, drop_last=True)
N = next(iter(train_loader))[0].shape[-1]

for data, target in train_loader:
    print(data.shape, target.shape)
    break


if __name__ == '__main__':
    model = ShallowCNN(N, channels=BATCH_SIZE)
    model.apply(lambda m: init_weights(m, method='he'))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= 1e-5)
    criterion = nn.MSELoss(reduction='none') #L1 / mean absolute error loss rather than MSEloss, for robustness against outliers
    min_test_loss, wait = float('inf'), 0

    wandb.init(project='CNN', config={'learning_rate': LEARNING_RATE, 'architecture': 'CNN', 'epochs': N_EPOCHS})

    train_losses, test_losses = [], []
    for epoch in range(N_EPOCHS):
        print(f"{epoch} / {N_EPOCHS}")
        batch_losses = []  # Initialize an empty list to hold batch losses for each epoch

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            target = torch.nan_to_num(target, nan=-1.0)

            mask = (target >= 0).float()
            loss = criterion(output, target)
            masked_loss = (loss*mask).mean() ##DON't MESS WITH THIS

            batch_losses.append(masked_loss.item())  # Appending the detached masked loss to batch_losses list
            masked_loss.backward()
            optimizer.step()

            
        train_losses.append(batch_losses)  # Appending the batch_losses list to train_losses list
        
        with torch.no_grad():  # Disable gradient computation
            test_batch_losses = []
            for test_batch_idx, (test_data, test_target) in enumerate(test_loader):
                
                test_output = model(test_data)
                
                test_target = torch.nan_to_num(test_target, nan=-1.0)
                
                test_mask = (test_target >= 0).float()
                test_loss = criterion(test_output, test_target)
                
                test_masked_loss = (test_loss * test_mask).sum() / test_mask.sum()
                test_batch_losses.append(test_masked_loss.item())
                
            test_epoch_loss = np.mean(test_batch_losses)
            test_losses.append(test_epoch_loss)
            print(f"testidation Loss: {test_epoch_loss}")
      
        if test_epoch_loss < min_test_loss:
            min_test_loss = test_epoch_loss
            wait = 0
            # Save the best model state
            torch.save(model.state_dict(), f'modelStateDict_{N_EPOCHS}_shallow.pth')
        elif EARLY_STOPPING:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping due to no improvement.")
                break
        wandb.log({'batch_loss': np.mean(batch_losses),'test_loss':test_epoch_loss})
    mean_train_losses = [np.mean(x) for x in train_losses]
    mean_test_losses = test_losses
    std_losses = [np.std(x) for x in train_losses]
    epochs = range(len(mean_train_losses))

    plt.plot(epochs, mean_train_losses, label='Train Loss')
    plt.plot(epochs, mean_test_losses, label='Validation Loss')
    if EARLY_STOPPING:
        plt.vlines(x=len(mean_train_losses)-PATIENCE, ymin=0, ymax=max(mean_train_losses), colors='red', label='Best Model')
    plt.title('Mean Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.ylim(bottom=0)

    plt.savefig(f'figures/CNN_train_val_losses.png')
