import numpy as np
from utils.binning import bin_spectra_with_id
from utils.preprocess import detrend_parallel, snv_parallel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import matplotlib.pyplot as plt




class SpectralDataset(Dataset):
    def __init__(self, X, y, skip_normalization=False):
        self.X, self.y = X.to_numpy().astype(np.float32), y.to_numpy().astype(np.float32)
        self.wl = np.array(X.columns).astype(np.float32)
        self.response_names = y.columns
        # Compute min and max for y for min-max scaling
        self.y_min = np.nanmin(self.y, axis = 0)
        self.y_max = np.nanmax(self.y, axis = 0)

        # Normalize y
        if skip_normalization:
            None
        else:
            self.y = (self.y - self.y_min) / (self.y_max - self.y_min)

        self.X = self.preprocess(self.X)

    # def preprocess(self, X):
    #     processed_X = X[:, np.where(self.wl >= 1100)[0]]
    #     snv_parallel(processed_X)
    #     detrend_parallel(processed_X)
    #     processed_X -= np.mean(processed_X, axis=1, keepdims=True)
    #     scaler = StandardScaler()
    #     processed_X = bin_spectra_with_id(scaler.fit_transform(processed_X), bin_width=10) #Optisk opløsning på apparatet er ca. 10 nm, en måling for hver .5 nm
    #     return processed_X


    def preprocess(self, X):
        spectra_stages = []

        # Initial Stage
        processed_X = X[:, np.where(self.wl >= 1100)[0]]
        spectra_stages.append(('After Cropping', processed_X.copy()))

        # After SNV
        snv_parallel(processed_X)
        spectra_stages.append(('After SNV', processed_X.copy()))

        # After Detrending
        detrend_parallel(processed_X)
        spectra_stages.append(('After Detrending', processed_X.copy()))

        # After Mean Subtraction
        processed_X -= np.mean(processed_X, axis=1, keepdims=True)
        spectra_stages.append(('After Mean Subtraction', processed_X.copy()))

        # After Standard Scaling and Binning
        scaler = StandardScaler()
        processed_X = bin_spectra_with_id(scaler.fit_transform(processed_X), bin_width=10)
        spectra_stages.append(('After Scaling and Binning', processed_X.copy()))

        #self.plot_faceted_spectra(spectra_stages)
        return processed_X


    # def plot_faceted_spectra(self, spectra_stages):
    #     num_stages = len(spectra_stages)
    #     plt.figure(figsize=(15, num_stages * 3))

    #     for i, (title, spectra) in enumerate(spectra_stages):
    #         plt.subplot(num_stages, 1, i + 1)
    #         for spectrum in spectra:
    #             plt.plot(spectrum, lw=0.5)  # Adjust line width as needed
    #         plt.title(title)
    #         plt.xlabel('Wavelength (nm)')
    #         plt.ylabel('Intensity')

    #     plt.tight_layout()
    #     plt.savefig(f'figures/preprocessing.png')


    def single_feature(self, feature):
                # Check if feature is specified by index or name
        if isinstance(feature, int):
            feature_idx = feature
        elif isinstance(feature, str) and feature in self.response_names:
            feature_idx = list(self.response_names).index(feature)
        else:
            raise ValueError("Feature must be an integer index or a valid feature name.")

        # Create new dataset with X and the selected feature from y
        X_new = self.X
        y_new = self.y[:, feature_idx].reshape(-1, 1)

        return X_new, y_new

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = self.X[idx], self.y[idx, 0:]
        return item
    
    def inverse_transform_y(self, y_normalized):
        # Denormalize y
        return y_normalized * (self.y_max - self.y_min) + self.y_min

