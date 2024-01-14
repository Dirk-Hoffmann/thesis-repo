#%% 
import datetime
from copy import copy
from os.path import exists

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import lightgbm as lgb
from colorspacious import cspace_converter
from joblib import Parallel, delayed, dump, load
from scipy import signal
from sklearn import decomposition, metrics, model_selection
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from utils.pls_kernel import pls
from utils.preprocess import detrend_parallel, snv_parallel
from utils.mycroft_utils import rmse, sep, bias, optlv
from utils.binning import bin_spectra_with_id

fileID = input('Enter file_id: ')

foss_orange = "#F58C28"
foss_labtexc = "#55A0D2"
foss_digital = "#00418C"
foss_blue = "#00285A"
foss_gray = "#999"

cmap = plt.cm.colors.LinearSegmentedColormap.from_list('foss', [(0,0.1568627450980392,0.35294117647058826),(0.9607843137254902,0.5490196078431373,0.1568627450980392)])
#%% Load data

X_train = pd.read_csv(f'data/x_{fileID}_300.csv').set_index('sampleid')
Y_train = pd.read_csv(f'data/y_{fileID}_300.csv').set_index('sampleid')



X_val = pd.read_csv(f'data/x_{fileID}.csv').set_index('sampleid')
Y_val = pd.read_csv(f'data/y_{fileID}.csv').set_index('sampleid')


wl = np.array(X_train.columns).astype(np.float32)
#%% 
# Plot raw data, color accoring to reference value
#

cdx = Y_train.iloc[:,0].to_numpy()

cl_norm = mpl.colors.Normalize(vmin=np.nanmin(cdx), vmax=np.nanmax(cdx))
m = cm.ScalarMappable(norm=cl_norm, cmap=cmap)

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(29.7/2.54, 21/2.54)

# Plot light colors in front (!) to make it look nicer
lab = cspace_converter("sRGB1", "CAM02-UCS")(m.to_rgba(cdx)[:,:3])
plt_ix = np.argsort(lab[:,0])

for n in plt_ix:
   _ = ax.plot(wl, X_train.iloc[n], color=[ *m.to_rgba(cdx[n])[0:3], *[0.15]], linewidth=2)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))
ax.grid()
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('-log(1/R)')
ax.set_title('Raw data')

cbar = fig.colorbar(cm.ScalarMappable(norm=cl_norm, cmap=cmap), ax=ax)
cbar.set_label('Reference', rotation=270)
plt.savefig(f'figures/{fileID}_rawdata.png')

#%% Preprocess the data

XP = copy(X_train.iloc[:,np.where(wl >= 1100)[0]].to_numpy())
YP = copy(Y_train.to_numpy())

XP_val = copy(X_val.iloc[:,np.where(wl >= 1100)[0]].to_numpy())
YP_val = copy(Y_val.to_numpy())

wlp = wl[np.where(wl >= 1100)[0]]

detrend_parallel(XP)
snv_parallel(XP)
detrend_parallel(XP_val)
snv_parallel(XP_val)


#%% 
# Plot preprocessed data, color accoring to reference value
#

cdx = YP[:,0]

cl_norm = mpl.colors.Normalize(vmin=np.nanmin(cdx), vmax=np.nanmax(cdx))
m = cm.ScalarMappable(norm=cl_norm, cmap=cmap)

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(29.7/2.54, 21/2.54)

# Plot light colors in front (!) to make it look nicer
lab = cspace_converter("sRGB1", "CAM02-UCS")(m.to_rgba(cdx)[:,:3])
plt_ix = np.argsort(lab[:,0])

for n in plt_ix:
   _ = ax.plot(wlp, XP[n,:], color=[ *m.to_rgba(cdx[n])[0:3], *[0.15]], linewidth=2)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))
ax.grid()
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('-log(1/R), Preprocessed')
ax.set_title('Preprocessed data')

cbar = fig.colorbar(cm.ScalarMappable(norm=cl_norm, cmap=cmap), ax=ax)
cbar.set_label('Reference', rotation=270)
plt.show()
plt.savefig(f'figures/{fileID}_preprocess.png')

#%% Mean center

XP = bin_spectra_with_id(XP,25)
XP_val = bin_spectra_with_id(XP_val,25)

XP_mean = np.mean(XP, axis=0)
YP_mean = np.mean(YP, axis=0)
XP_val_mean = np.mean(XP_val, axis= 0)
YP_val_mean = np.mean(YP_val, axis= 0)


XP-=XP_mean
YP-=YP_mean
XP_val -= XP_val_mean
YP_val -= YP_val_mean

# Calculate standard deviation for YP and YP_val
std_YP = np.std(YP, axis=0)
std_YP_val = np.std(YP_val, axis=0)

# Create filters based only on Y-values
filter_YP = np.all(abs(YP) < 5 * std_YP, axis=1)
filter_YP_val = np.all(abs(YP_val) < 5 * std_YP_val, axis=1)

# Apply the filters to both X and Y datasets to ensure consistent lengths
XP = XP[filter_YP]
YP = YP[filter_YP]
XP_val = XP_val[filter_YP_val]
YP_val = YP_val[filter_YP_val]


##### time for KNN

neighbors = int(input('Enter number of neighbors: '))

### First - K-folds

predictions = np.zeros(len(YP))

kf = KFold(n_splits=7, shuffle=True)

### opdater med cosinus distance eller correlation distance / pearson distance

for train_index, test_index in kf.split(XP):
    XP_train, XP_test = XP[train_index], XP[test_index]
    YP_train, YP_test = YP[train_index], YP[test_index]

    # Feature scaling
    scaler = StandardScaler()
    XP_train_scaled = scaler.fit_transform(XP_train)
    XP_test_scaled = scaler.transform(XP_test)

    
    # Create KNN regressor
    knn = KNeighborsRegressor(n_neighbors=neighbors, metric='correlation')

    # Fit model
    knn.fit(XP_train_scaled, YP_train)


    predictions[test_index] = knn.predict(XP_test_scaled).flatten()

rmsecv = np.sqrt(mean_squared_error(YP, predictions))


#### Next - RMSEC & RMSEP

scaler = StandardScaler()
XP_scaled = scaler.fit_transform(XP)
XP_val_scaled = scaler.fit_transform(XP_val)

knn = KNeighborsRegressor(n_neighbors=neighbors, metric='correlation')

knn.fit(XP_scaled, YP)

RMSEC_predictions = knn.predict(XP_scaled).flatten()
RMSEP_predictions = knn.predict(XP_val_scaled).flatten()

rmsec = np.sqrt(mean_squared_error(YP, RMSEC_predictions))
rmsep = np.sqrt(mean_squared_error(YP_val, RMSEP_predictions))

print(f"RMSECV KNN K-fold: {rmsecv}\n"
      f"RMSEP KNN: {rmsep}\n"
      f"RMSEC KNN: {rmsec}\n")


plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(29.7/2.54, 21/2.54)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].axis('equal')
axs[1].axis('equal')
axs[2].axis('equal')
axs[0].axline((0, 0), slope=1)
axs[1].axline((0, 0), slope=1)
axs[2].axline((0, 0), slope=1)

axs[0].scatter(YP+YP_mean, RMSEC_predictions+YP_mean, facecolors='none', edgecolors='r')
axs[0].set_title(f'Calibration | RMSEC : {rmsec}')

axs[1].scatter(YP+YP_mean, predictions+YP_mean, facecolors='none', edgecolors='b')
axs[1].set_title(f'Cross-Validation | RMSECV : {rmsecv}')

axs[2].scatter(YP_val+YP_val_mean, RMSEP_predictions+YP_val_mean, facecolors='none', edgecolors='g')
axs[2].set_title(f'Validation | RMSEP : {rmsep}')




# plt.scatter(YP+YP_mean, predictions+YP_mean, label = 'Cross-Validation')
# plt.scatter(YP_val+YP_val_mean, RMSEP_predictions+YP_val_mean, label = 'Validation')
# plt.scatter(YP+YP_mean, RMSEC_predictions+YP_mean, label = 'Calibration')

ax.xaxis.set_major_locator(plt.MaxNLocator(7))
ax.grid()
ax.set_xlabel('Measured')
ax.set_ylabel('CV Predicted')
fig.suptitle(f'Actual vs. predicted | KNN neighbors={neighbors}')


plt.savefig(f'figures/{fileID}_KNN_kfold.png')