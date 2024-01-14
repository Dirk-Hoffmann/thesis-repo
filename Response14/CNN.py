# Hyperparameters
BATCH_SIZE = 512
N_EPOCHS = 40
LEARNING_RATE = 0.001
TEST_SIZE = 0.25
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
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utils.datasplitter import train_dataset, test_dataset, val_dataset, cmap
from utils.classes import SpectralDataset


class MultiOutputCNN(nn.Module):
    def __init__(self):
        super(MultiOutputCNN, self).__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(1, 128, 3)
        self.conv2 = nn.Conv1d(128, 2048, 3)
        self.conv3 = nn.Conv1d(2048, 128, 10)
        self.fc1 = nn.Linear(16384, 256)  # Assumes three Conv1D and one MaxPool1D layer
        self.fc2 = nn.Linear(256, 4096)
        self.fc3 = nn.Linear(4096, 14)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x) # Concentrations can't be negative
        return x

    # def forward(self, x):
    #     x = nn.ReLU()(self.conv1(x))
    #     print("After conv1:", x.shape)

    #     x = self.pool(x)
    #     print("After pooling:", x.shape)

    #     x = nn.ReLU()(self.conv2(x))
    #     print("After conv2:", x.shape)

    #     x = nn.ReLU()(self.conv3(x))
    #     print("After conv3:", x.shape)

    #     x = x.view(x.size(0), -1)
    #     print("After flattening:", x.shape)

    #     x = nn.ReLU()(self.fc1(x))
    #     print("After fc1:", x.shape)

    #     x = nn.ReLU()(self.fc2(x))
    #     print("After fc2:", x.shape)

    #     x = self.dropout(x)
    #     print("After dropout:", x.shape)

    #     x = self.fc3(x)  # Concentrations can't be negative
    #     print("After fc3 (final output):", x.shape)

    #     return x
def init_weights(m, method='he'):
    init_fn = nn.init.xavier_uniform_ if method == 'xavier' else nn.init.kaiming_uniform_
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init_fn(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# cmap_list = []
# amap_list = []
# print(X_val)
# for i in X_val['2200.0']:
#     if i > 2:
#         cmap_list.append('red')
#     else:
#         cmap_list.append('blue')

# amap = np.array(amap_list)
# cmap = np.array(cmap_list)


#print(train_dataset.response_names)



# for data, target in train_loader:
#     print(data.shape, target.shape)
#     break


def run_CNN(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, name = ''):

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle= False, drop_last=True)
    N = next(iter(train_loader))[0].shape[-1]

    model = MultiOutputCNN()
    model.apply(lambda m: init_weights(m, method='he'))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= 1e-5)
    criterion = nn.MSELoss(reduction='none') #L1 / mean absolute error loss rather than MSEloss, for robustness against outliers
    min_test_loss, wait = float('inf'), 0

  #  wandb.init(project='CNN', config={'learning_rate': LEARNING_RATE, 'architecture': 'CNN', 'epochs': N_EPOCHS})

    train_losses, test_losses = [], []

    for epoch in range(N_EPOCHS):
        print(f"Deep CNN progress: epoch {epoch} / {N_EPOCHS} Epochs")
        batch_losses = []  # Initialize an empty list to hold batch losses for each epoch

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.unsqueeze(1) #Adds a channel dimension to data

            output = model(data)
            target = torch.nan_to_num(target, nan=-1.0)

            mask = (target >= 0).float()
            loss = criterion(output, target)
            masked_loss = (loss*mask).mean()

            batch_losses.append(masked_loss.item())  # Appending the detached masked loss to batch_losses list
            masked_loss.backward()
            optimizer.step()

            
        train_losses.append(batch_losses)  # Appending the batch_losses list to train_losses list
         
        with torch.no_grad():  # Disable gradient computation
            test_batch_losses = []
            for test_batch_idx, (test_data, test_target) in enumerate(test_loader):
                test_data = test_data.unsqueeze(1)
                test_output = model(test_data)
                
                test_target = torch.nan_to_num(test_target, nan=-1.0)
                
                test_mask = (test_target >= 0).float()
                test_loss = criterion(test_output, test_target)
                
                test_masked_loss = (test_loss * test_mask).sum() / test_mask.sum()
                test_batch_losses.append(test_masked_loss.item())
                
            test_epoch_loss = np.mean(test_batch_losses)
            test_losses.append(test_epoch_loss)
            print(f"Deep CNN Test Loss: {test_epoch_loss}")
      
        if test_epoch_loss < min_test_loss:
            min_test_loss = test_epoch_loss
            wait = 0
            # Save the best model state
            torch.save(model.state_dict(), f'state_dicts/10_runs/modelStateDict_{N_EPOCHS}_deep_{name}.pth')
        elif EARLY_STOPPING:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping due to no improvement.")
                break
      #  wandb.log({'batch_loss': np.mean(batch_losses),'test_loss':test_epoch_loss})
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

    return val_loader


if __name__ == '__main__':
    run_CNN()