import sys

# Hyperparameters
BATCH_SIZE = 512
N_EPOCHS = 1000
HIDDEN_LAYER_SIZE = 128
LEARNING_RATE = 0.001
TEST_SIZE = 0.4
RANDOM_STATE = 42
PATIENCE = N_EPOCHS // 10
EARLY_STOPPING = True
BOTTLENECK_SIZE = 20 #Only for autoencoder NN
PCA_SIZE = 20 #Only for linear 

from utils.mycroft_utils import rmse, sep, bias, optlv
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
from copy import copy
from joblib import load
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.classes import SpectralDataset
from utils.datasplitter import train_dataset, test_dataset, val_dataset



class ShallowCNN(nn.Module):
    def __init__(self, N):
        super(ShallowCNN, self).__init__()

        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(1, 1, 3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(140, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, 1)


    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
       # x = self.dropout(x)
        x = self.fc2(x)
        print(x.shape)
        return x

import torch.nn as nn

class ShallowCNN_Linear(nn.Module):
    def __init__(self, N):
        super(ShallowCNN, self).__init__()

        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(1, 1, 3, padding=1)
        self.dropout = nn.Dropout(0.5)

        # Linear layer to mimic PCA transformation
        self.pca = nn.Linear(140, PCA_SIZE)

        self.fc1 = nn.Linear(PCA_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, 1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Apply the PCA-like linear transformation
        x = self.pca(x)

        x = nn.ReLU()(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


class ShallowCNN_Autoencoder(nn.Module):
    def __init__(self, N):
        super(ShallowCNN_Autoencoder, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.encoder_pool = nn.MaxPool1d(2)
        self.encoder_fc1 = nn.Linear(N // 2 * 16, HIDDEN_LAYER_SIZE)
        self.bottleneck = nn.Linear(HIDDEN_LAYER_SIZE, BOTTLENECK_SIZE)

        # Decoder
        self.decoder_fc1 = nn.Linear(BOTTLENECK_SIZE, HIDDEN_LAYER_SIZE)
        self.decoder_fc2 = nn.Linear(HIDDEN_LAYER_SIZE, N // 2 * 16)
        self.decoder_unpool = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_conv1 = nn.Conv1d(16, 1, 3, padding=1)
        self.decoder_fc3 = nn.Linear(280,1)

    def forward(self, x):
        # Encoding
        x = nn.ReLU()(self.encoder_conv1(x))
        x = self.encoder_pool(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.encoder_fc1(x))
        x = nn.ReLU()(self.bottleneck(x))

        # Decoding
        x = nn.ReLU()(self.decoder_fc1(x))
        x = nn.ReLU()(self.decoder_fc2(x))
        x = x.view(x.size(0), 16, -1)
        x = self.decoder_unpool(x)
        x = self.decoder_conv1(x)
        x = nn.ReLU()(self.decoder_fc3(x))
        x = x.view(x.size(0), -1)

       # print(x.shape)

        return x

    def encode(self, x):
        # Function to extract features from the bottleneck layer
        x = nn.ReLU()(self.encoder_conv1(x))
        x = self.encoder_pool(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.encoder_fc1(x))
        return self.bottleneck(x)

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

#print(train_dataset.response_names)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle= False, drop_last=True)
N = next(iter(train_loader))[0].shape[-1]

for data, target in train_loader:
    print(data.shape, target.shape)
    break


if __name__ == '__main__':
    N_features = y_train.shape[1]  # Number of output features
    models = []

    for feature_idx in range(N_features):
        print(f"Training model for feature {feature_idx}")
        model = ShallowCNN_Linear(N)
        model.apply(lambda m: init_weights(m, method='he'))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        criterion = nn.MSELoss(reduction='none')        
        train_losses, test_losses = [], []
        min_test_loss, wait = float('inf'), 0

       
        for epoch in range(N_EPOCHS):
            batch_losses = []  # Initialize an empty list to hold batch losses for each epoch
            for batch_idx, (data, target) in enumerate(train_loader):
                feature_target = target[:, feature_idx].unsqueeze(1)  # Select the specific feature
                data = data.unsqueeze(1) #Adds a channel dimension to data
                optimizer.zero_grad()
                output = model(data)
                feature_target = torch.nan_to_num(feature_target, nan=-1.0)

                mask = (feature_target >= 0).float()
                loss = criterion(output, feature_target)
                masked_loss = (loss*mask).mean() ##DON't MESS WITH THIS

                batch_losses.append(masked_loss.item())  # Appending the detached masked loss to batch_losses list
                masked_loss.backward()
                optimizer.step()

                
            train_losses.append(np.mean(batch_losses))  # Appending the batch_losses list to train_losses list
            
            with torch.no_grad():  # Disable gradient computation
                test_batch_losses = []
                for test_batch_idx, (test_data, test_target) in enumerate(test_loader):
                    test_data = test_data.unsqueeze(1)
                    test_output = model(test_data)
                    
                    test_feature_target = test_target[:, feature_idx].unsqueeze(1)
                    test_target = torch.nan_to_num(test_target, nan=-1.0)
                    
                    test_mask = (test_target >= 0).float()
                    test_loss = criterion(test_output, test_target)
                    
                    test_masked_loss = (test_loss * test_mask).sum() / test_mask.sum()
                    test_batch_losses.append(test_masked_loss.item())
                    
                test_epoch_loss = np.mean(test_batch_losses)
                test_losses.append(test_epoch_loss)
                #print(f"testidation Loss: {test_epoch_loss}")
        
            if test_epoch_loss < min_test_loss:
                min_test_loss = test_epoch_loss
                wait = 0
                # Save the best model state
                bestmodel = model
                torch.save(model.state_dict(), f'modelStateDict_feature_{feature_idx}.pth')
            elif EARLY_STOPPING:
                wait += 1
                if wait >= PATIENCE:
                    print("Early stopping due to no improvement.")
                    break 
        
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
        models.append(bestmodel)

    # Plotting section
    plt.figure(figsize=(10, 10))
    plt.suptitle(f'Shallow CNN, Single output, Hidden Layers = {HIDDEN_LAYER_SIZE}')
    csv_output = ["ReLU", BATCH_SIZE, HIDDEN_LAYER_SIZE]
    print(len(models))
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(f'modelStateDict_feature_{i}.pth'))
        model.eval()

        true_vals = []
        pred_vals = []

        # Forward pass on validation set
        with torch.no_grad():
            for data, target in val_loader:
                data = data.unsqueeze(1)  # Adding channel dimension to data
                output = model(data)
                true_vals.append(target.numpy())
                pred_vals.append(output.numpy())

        # Convert list of arrays to single NumPy arrays
        true_vals = val_dataset.inverse_transform_y(np.concatenate(true_vals))
        pred_vals = train_dataset.inverse_transform_y(np.concatenate(pred_vals))

        print("True values shape:", true_vals.shape)
        print("Predicted values shape:", pred_vals.shape)


        # Create mask for non-missing values and exclude values outside of 3 standard deviations
        mask = ~np.isnan(true_vals[:, i]) & ~np.isnan(pred_vals[:, i])
        true_mean, true_std = np.nanmean(true_vals[:, i]), np.nanstd(true_vals[:, i])
        pred_mean, pred_std = np.nanmean(pred_vals[:, i]), np.nanstd(pred_vals[:, i])
        mask &= (true_vals[:, i] > true_mean - 3*true_std) & (true_vals[:, i] < true_mean + 3*true_std)
        mask &= (pred_vals[:, i] > pred_mean - 3*pred_std) & (pred_vals[:, i] < pred_mean + 3*pred_std)

        rounded_rmse = "{:.3f}".format(rmse(true_vals[mask, i], pred_vals[mask, i])[0])
        print(rounded_rmse, N_EPOCHS)
        csv_output.append(rounded_rmse)

        plt.subplot(4, 4, i+1)
        plt.scatter(true_vals[mask, i], pred_vals[mask, i], alpha=0.5, facecolors='none', edgecolors='blue')
        limits = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(limits, limits, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        plt.xlim(limits)
        plt.ylim(limits)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{val_dataset.response_names[i]} \nRMSE = {rounded_rmse}', loc='left')

    plt.tight_layout()
    plt.savefig(f'figures/shallowCNN_autoencoder_SingleOutput_{HIDDEN_LAYER_SIZE}_HL_ReLU.png')

    with open('shallowCNNLosses.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_output)
