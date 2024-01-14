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
from colorspacious import cspace_converter
from joblib import Parallel, delayed, dump, load
from scipy import signal
from sklearn import decomposition, metrics, model_selection

from utils.pls_kernel import pls
from utils.preprocess import detrend_parallel, snv_parallel
from utils.mycroft_utils import rmse, sep, bias, optlv

foss_orange = "#F58C28"
foss_labtexc = "#55A0D2"
foss_digital = "#00418C"
foss_blue = "#00285A"
foss_gray = "#999"

cmap = plt.cm.colors.LinearSegmentedColormap.from_list('foss', [(0,0.1568627450980392,0.35294117647058826),(0.9607843137254902,0.5490196078431373,0.1568627450980392)])
#%% Load data

X = pd.read_csv('x_protein.csv').set_index('sampleid')
Y = pd.read_csv('y_protein.csv').set_index('sampleid')

wl = np.array(X.columns).astype(np.float32)
#%% 
# Plot raw data, color accoring to reference value
#

cdx = Y.iloc[:,0].to_numpy()

cl_norm = mpl.colors.Normalize(vmin=np.nanmin(cdx), vmax=np.nanmax(cdx))
m = cm.ScalarMappable(norm=cl_norm, cmap=cmap)

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(29.7/2.54, 21/2.54)

# Plot light colors in front (!) to make it look nicer
lab = cspace_converter("sRGB1", "CAM02-UCS")(m.to_rgba(cdx)[:,:3])
plt_ix = np.argsort(lab[:,0])

for n in plt_ix:
   _ = ax.plot(wl, X.iloc[n], color=[ *m.to_rgba(cdx[n])[0:3], *[0.15]], linewidth=2)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))
ax.grid()
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('-log(1/R)')
ax.set_title('Raw data')

cbar = fig.colorbar(cm.ScalarMappable(norm=cl_norm, cmap=cmap), ax=ax)
cbar.set_label('Reference', rotation=270)
plt.show()

#%% Preprocess the data

XP = copy(X.iloc[:,np.where(wl >= 1100)[0]].to_numpy())
YP = copy(Y.to_numpy())
wlp = wl[np.where(wl >= 1100)[0]]

detrend_parallel(XP)
snv_parallel(XP)

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

#%% Mean center

XP_mean = np.mean(XP, axis=0)
YP_mean = np.mean(YP, axis=0)

XP-=XP_mean
YP-=YP_mean

#%% Do some PLS

#cv = model_selection.KFold(n_splits=5, shuffle=True)
cv = model_selection.LeaveOneOut()
M = pls(XP, YP, None, 20, cv)

rmsec = rmse(YP, M['pred']['cal'])
rmsecv = rmse(YP, M['pred']['cv'])

#comps = optlv(rmsecv)
comps = np.where(rmsecv == np.min(rmsecv))[0]

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(29.7/2.54, 21/2.54)

plt.scatter(YP+YP_mean, M['pred']['cv'][:,comps]+YP_mean)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))
ax.grid()
ax.set_xlabel('Measured')
ax.set_ylabel('CV Predicted')
ax.set_title('Actual vs. predicted')

plt.show()



