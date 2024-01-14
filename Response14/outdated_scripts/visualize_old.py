#%% 

def main(X, Y, fileID):
   import datetime
   from copy import copy
   from os.path import exists

   import matplotlib as mpl
   import matplotlib.cm as cm
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import lightgbm as lgb
   import progressbar
   import csv
   from colorspacious import cspace_converter
   from joblib import Parallel, delayed, dump, load
   from scipy import signal
   from sklearn import decomposition, metrics, model_selection
   from sklearn.linear_model import LassoCV
   from sklearn.preprocessing import MinMaxScaler
   from utils.lasso_regression import lasso
   from sklearn.model_selection import KFold, train_test_split
   from sklearn.metrics import mean_squared_error
   from sklearn.neighbors import KNeighborsRegressor
   from utils.pls_kernel import pls
   from utils.binning import bin_spectra_with_id
   from utils.preprocess import detrend_parallel, snv_parallel
   from utils.mycroft_utils import rmse, sep, bias, optlv
   from utils.classes import SpectralDataset
   from utils.datasplitter import train_dataset, test_dataset, val_dataset




   run_PLS = True
   run_Lasso = True
   run_LGB = True
   run_KNN = True
   #fileID = input('Enter file_id: ')


   # foss_orange = "#F58C28"
   # foss_labtexc = "#55A0D2"
   # foss_digital = "#00418C"
   # foss_blue = "#00285A"
   # foss_gray = "#999"

   # cmap = plt.cm.colors.LinearSegmentedColormap.from_list('foss', [(0,0.1568627450980392,0.35294117647058826),(0.9607843137254902,0.5490196078431373,0.1568627450980392)])
   # #%% Load data
   # # FILTER FOR MISSING VALUES BEFORE TURNING INTO SPLITS
   # # Remove rows in Y with NaN and get valid indices
   # valid_row_positions = np.where(~Y.isna())[0]
   # Y = Y.iloc[valid_row_positions]
   # X = X.iloc[valid_row_positions]


   # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=42)  


   # # X_train = pd.read_csv(f'data/x_{fileID}_300.csv').set_index('sampleid')
   # # Y_train = pd.read_csv(f'data/y_{fileID}_300.csv').set_index('sampleid')

   # # X_val = pd.read_csv(f'data/x_{fileID}.csv').set_index('sampleid')
   # # Y_val = pd.read_csv(f'data/y_{fileID}.csv').set_index('sampleid')


   # wl = np.array(X_train.columns).astype(np.float32)
   # #%% 
   # # Plot raw data, color accoring to reference value
   # #

   # cdx = Y_train.iloc[:,0].to_numpy()

   # cl_norm = mpl.colors.Normalize(vmin=np.nanmin(cdx), vmax=np.nanmax(cdx))
   # m = cm.ScalarMappable(norm=cl_norm, cmap=cmap)

   # plt.rcParams.update({'font.size': 8})
   # fig, ax = plt.subplots(1, 1)
   # fig.set_size_inches(29.7/2.54, 21/2.54)

   # # Plot light colors in front (!) to make it look nicer
   # lab = cspace_converter("sRGB1", "CAM02-UCS")(m.to_rgba(cdx)[:,:3])
   # plt_ix = np.argsort(lab[:,0])

   # for n in plt_ix:
   #    _ = ax.plot(wl, X_train.iloc[n], color=[ *m.to_rgba(cdx[n])[0:3], *[0.15]], linewidth=2)

   # ax.xaxis.set_major_locator(plt.MaxNLocator(7))
   # ax.grid()
   # ax.set_xlabel('Wavelength [nm]')
   # ax.set_ylabel('-log(1/R)')
   # ax.set_title('Raw data')

   # cbar = fig.colorbar(cm.ScalarMappable(norm=cl_norm, cmap=cmap), ax=ax)
   # cbar.set_label('Reference', rotation=270)
   # # plt.savefig(f'figures/{fileID}_rawdata.png')

   # #%% Preprocess the data

   # XP = copy(X_train.iloc[:,np.where(wl >= 1100)[0]].to_numpy())
   # YP = copy(Y_train.to_numpy())

   # XP_val = copy(X_val.iloc[:,np.where(wl >= 1100)[0]].to_numpy())
   # YP_val = copy(Y_val.to_numpy())

   # wlp = wl[np.where(wl >= 1100)[0]]

   # detrend_parallel(XP)
   # snv_parallel(XP)
   # detrend_parallel(XP_val)
   # snv_parallel(XP_val)

   # #%% 
   # # Plot preprocessed data, color accoring to reference value
   # #

   # cdx = YP[:,0]

   # cl_norm = mpl.colors.Normalize(vmin=np.nanmin(cdx), vmax=np.nanmax(cdx))
   # m = cm.ScalarMappable(norm=cl_norm, cmap=cmap)

   # plt.rcParams.update({'font.size': 8})
   # fig, ax = plt.subplots(1, 1)
   # fig.set_size_inches(29.7/2.54, 21/2.54)

   # # Plot light colors in front (!) to make it look nicer
   # lab = cspace_converter("sRGB1", "CAM02-UCS")(m.to_rgba(cdx)[:,:3])
   # plt_ix = np.argsort(lab[:,0])

   # for n in plt_ix:
   #    _ = ax.plot(wlp, XP[n,:], color=[ *m.to_rgba(cdx[n])[0:3], *[0.15]], linewidth=2)

   # ax.xaxis.set_major_locator(plt.MaxNLocator(7))
   # ax.grid()
   # ax.set_xlabel('Wavelength [nm]')
   # ax.set_ylabel('-log(1/R), Preprocessed')
   # ax.set_title('Preprocessed data')

   # cbar = fig.colorbar(cm.ScalarMappable(norm=cl_norm, cmap=cmap), ax=ax)
   # cbar.set_label('Reference', rotation=270)
   # plt.show()
   # plt.savefig(f'figures/{fileID}_preprocess.png')


   # #%% Mean center
   # XP = bin_spectra_with_id(XP,25)
   # XP_val = bin_spectra_with_id(XP_val,25)

   # XP_mean = np.mean(XP, axis=0)
   # YP_mean = np.mean(YP, axis=0)



   # XP-=XP_mean
   # YP-=YP_mean
   # XP_val -= XP_mean
   # YP_val -= YP_mean
   # ##### Brug calibration means til at normalisere validerings set.

   # # Calculate standard deviation for YP and YP_val
   # std_YP = np.std(YP, axis=0)
   # std_YP_val = np.std(YP_val, axis=0)

   # # Create filters based only on Y-values
   # filter_YP = np.all(abs(YP) < 3 * std_YP, axis=1)
   # filter_YP_val = np.all(abs(YP_val) < 3 * std_YP_val, axis=1)


   # # Apply the filters to both X and Y datasets to ensure consistent lengths
   # XP = XP[filter_YP]
   # YP = YP[filter_YP]
   # XP_val = XP_val[filter_YP_val]
   # YP_val = YP_val[filter_YP_val]

   # print(XP,YP)

   
   ### HYPERPARAMETERS
   TEST_SIZE = .25
   RANDOM_STATE = 42
   run_PLS = True
   run_Lasso = True
   run_LGB = True
   run_KNN = True

      # FILTER FOR MISSING VALUES BEFORE TURNING INTO SPLITS
   # Remove rows in Y with NaN and get valid indices
   valid_row_positions = np.where(~Y.isna())[0]
   Y = Y.iloc[valid_row_positions]
   X = X.iloc[valid_row_positions]

   X, Y

   X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
   X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)
   

   cmap_list = []

   for i in X_val['2200.0']:
      if i > 2:
         cmap_list.append('red')
      else:
         cmap_list.append('blue')

   cmap = np.array(cmap_list)
   
   train_dataset = SpectralDataset(X_train, y_train, skip_normalization=True)
   test_dataset = SpectralDataset(X_test, y_test, skip_normalization=True)
   val_dataset = SpectralDataset(X_val, y_val, skip_normalization=True)

   XP, YP = train_dataset[:]

   YP_mean = np.mean(YP, axis=0)

   XP_test, YP_test = test_dataset[:]

   XP_val, YP_val = val_dataset[:]
  


   print(XP, YP, cmap)
   if run_PLS:
      #%% Do some PLS

      #cv = model_selection.KFold(n_splits=5, shuffle=True)
      cv = model_selection.KFold(n_splits=10, shuffle=True)
      M = pls(XP, YP, XP_val, 20, cv)

      rmsec = rmse(YP, M['pred']['cal'])
      rmsecv = rmse(YP, M['pred']['cv'])

      #comps = optlv(rmsecv)
      comps_cv = np.where(rmsecv == np.min(rmsecv))[0]
      comps_c = np.where(rmsec == np.min(rmsec))[0]


      #### Next - RMSEC & RMSEP


      rmsep = rmse(YP_val, M['pred']['val']+YP_mean)
      comps_p = np.where(rmsep == np.min(rmsep))[0]


      print(f"RMSECV PLS K-fold: {rmsecv}\n"
            f"RMSEP PLS: {rmsep}\n"
            f"RMSEC PLS: {rmsec}\n")

      print((YP+YP_mean).shape, (M['pred']['cv']+YP_mean).shape)

      # plt.scatter(YP+YP_mean, predictions+YP_mean, label = 'Cross-Validation')
      # plt.scatter(YP_val+YP_mean, RMSEP_predictions+YP_mean, label = 'Validation')
      # plt.scatter(YP+YP_mean, RMSEC_predictions+YP_mean, label = 'Calibration')

      # ax.xaxis.set_major_locator(plt.MaxNLocator(7))
      # ax.grid()
      # ax.set_xlabel('Measured')
      # ax.set_ylabel('CV Predicted')
      # fig.suptitle(f'Actual vs. predicted  {fileID}')

#      plt.savefig(f'figures/{fileID}_PLS.png')

   if run_Lasso:
      #%% do some lasso regression >:)
      cv = model_selection.KFold(n_splits=10, shuffle=True)

      lasso_cv = LassoCV(alphas=[0.0001,0.001,0.01,0.1, 0.2], cv=5)

      # Fit the model
      lasso_cv.fit(XP, YP)

      # Optimal alpha
      optimal_alpha = lasso_cv.alpha_
      print(f'Optimal Alpha: {optimal_alpha}')

      lasso_model = lasso(XP, YP, XP_val, cv, optimal_alpha)

      rmse_values = []

      rmsec_lasso = rmse(YP, lasso_model['pred']['cal'])
      rmsecv_lasso = rmse(YP, lasso_model['pred']['cv'])
      rmsep_lasso = np.round(rmse(YP_val, lasso_model['pred']['val']),3)


      comps = np.where(rmsecv == np.min(rmsecv))[0]

      # plt.rcParams.update({'font.size': 8})
      # fig, ax = plt.subplots(1, 1)
      # fig.set_size_inches(29.7/2.54, 21/2.54)

      # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
      # axs[0].axis('equal')
      # axs[1].axis('equal')
      # axs[2].axis('equal')
      # axs[0].axline((0, 0), slope=1)
      # axs[1].axline((0, 0), slope=1)
      # axs[2].axline((0, 0), slope=1)




      # plt.scatter(YP+YP_mean, predictions+YP_mean, label = 'Cross-Validation')
      # plt.scatter(YP_val+YP_val_mean, RMSEP_predictions+YP_val_mean, label = 'Validation')
      # plt.scatter(YP+YP_mean, RMSEC_predictions+YP_mean, label = 'Calibration')

      # ax.xaxis.set_major_locator(plt.MaxNLocator(7))
      # ax.grid()
      # ax.set_xlabel('Measured')
      # ax.set_ylabel('CV Predicted')
      # fig.suptitle(f'Actual vs. predicted {fileID}| Model = Lasso, alpha = {optimal_alpha}')


      # plt.savefig(f'figures/{fileID}_LASSO.png')

   if run_LGB:
      #%% time to build a decision tree regressor
      #%% as this will potentially be a bit more computationally heavy, we'll do a regular k-folds

      sort = False

      # Sort data based on YP values

      YP_squeezed = np.squeeze(YP)
      sorted_indices = np.argsort(YP_squeezed)

      XP_sorted = XP[sorted_indices]
      YP_sorted = YP[sorted_indices]

      # Initialize k-fold cross-validation

      kf = KFold(n_splits=7, shuffle=True)

      params = {'objective': 'regression',
               'metric': 'rmse',
               'boosting_type': 'gbdt',
               'max_depth':5,
               'verbosity': 0,
               'min_data_in_bin':15, 
               'min_data_in_leaf':15, 
               'force_col_wise':True,
               }

      # Placeholder for predictions
      predictions = np.zeros(len(YP_sorted))

      bst = None

      # Run k-fold cross-validation
      for train_index, test_index in kf.split(XP):
         XP_train, XP_test = XP[train_index], XP[test_index]
         YP_train, YP_test = YP[train_index], YP[test_index]

         train_data = lgb.Dataset(XP_train, label=YP_train)
         test_data = lgb.Dataset(XP_test, label=YP_test, reference=train_data)
            
         bst = lgb.train(params, train_data, num_boost_round=100, init_model=bst)
         predictions[test_index] = bst.predict(XP_test)


      rmse = np.sqrt(mean_squared_error(YP, predictions))


      rmsecv_lgb = np.sqrt(mean_squared_error(YP, predictions))


      #### Next - RMSEC & RMSEP
      lgb_train_XP = lgb.Dataset(XP, label=YP)

      bst = lgb.train(params, lgb_train_XP, num_boost_round=100, init_model=bst)

      RMSEC_predictions = bst.predict(XP)
      RMSEP_predictions = bst.predict(XP_val)

      rmsec_lgb = np.sqrt(mean_squared_error(YP, RMSEC_predictions))
      rmsep_lgb = np.sqrt(mean_squared_error(YP_val, RMSEP_predictions))


      # plt.rcParams.update({'font.size': 8})
      # fig, ax = plt.subplots(1, 1)
      # fig.set_size_inches(29.7/2.54, 21/2.54)


      # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
      # axs[0].axis('equal')
      # axs[1].axis('equal')
      # axs[2].axis('equal')
      # axs[0].axline((0, 0), slope=1)
      # axs[1].axline((0, 0), slope=1)
      # axs[2].axline((0, 0), slope=1)




      # plt.scatter(YP+YP_mean, predictions+YP_mean, label = 'Cross-Validation')
      # plt.scatter(YP_val+YP_val_mean, RMSEP_predictions+YP_val_mean, label = 'Validation')
      # plt.scatter(YP+YP_mean, RMSEC_predictions+YP_mean, label = 'Calibration')

      # ax.xaxis.set_major_locator(plt.MaxNLocator(7))
      # ax.grid()
      # ax.set_xlabel('Measured')
      # ax.set_ylabel('CV Predicted')
      # fig.suptitle(f'Actual vs. predicted {fileID} | Model = Light Gradient Boost')

      # plt.savefig(f'figures/{fileID}_LGB_kfold.png')

   if run_KNN:
   ##### time for KNN

      #neighbors = int(input('Enter number of neighbors: '))
      neighbors = 7

      ### First - K-folds

      predictions_knn = np.zeros(len(YP))

      kf = KFold(n_splits=7, shuffle=True)

      ### opdater med cosinus distance eller correlation distance / pearson distance

      for train_index, test_index in kf.split(XP):
         XP_train, XP_test = XP[train_index], XP[test_index]
         YP_train, YP_test = YP[train_index], YP[test_index]

         # Feature scaling
         #  scaler = StandardScaler()
         #  XP_train_scaled = scaler.fit_transform(XP_train)
         #  XP_test_scaled = scaler.transform(XP_test)

         
         # Create KNN regressor
         knn = KNeighborsRegressor(n_neighbors=neighbors, metric='correlation')

         # Fit model
         knn.fit(XP_train, YP_train)


         predictions_knn[test_index] = knn.predict(XP_test).flatten()

      rmsecv_knn = np.sqrt(mean_squared_error(YP, predictions_knn))


      #### Next - RMSEC & RMSEP

      # scaler = StandardScaler()
      # XP_scaled = scaler.fit_transform(XP)
      # XP_val_scaled = scaler.fit_transform(XP_val)

      knn = KNeighborsRegressor(n_neighbors=neighbors, metric='correlation')

      knn.fit(XP, YP)

      RMSEC_predictions_knn = knn.predict(XP).flatten()
      RMSEP_predictions_knn = knn.predict(XP_val).flatten()

      rmsec_knn = np.sqrt(mean_squared_error(YP, RMSEC_predictions_knn))
      rmsep_knn = np.round(np.sqrt(mean_squared_error(YP_val, RMSEP_predictions_knn)),3)



      # plt.rcParams.update({'font.size': 8})
      # fig, ax = plt.subplots(1, 1)
      # fig.set_size_inches(29.7/2.54, 21/2.54)

      # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
      # axs[0].axis('equal')
      # axs[1].axis('equal')
      # axs[2].axis('equal')
      # axs[0].axline((0, 0), slope=1)
      # axs[1].axline((0, 0), slope=1)
      # axs[2].axline((0, 0), slope=1)




      # plt.scatter(YP+YP_mean, predictions+YP_mean, label = 'Cross-Validation')
      # plt.scatter(YP_val+YP_val_mean, RMSEP_predictions+YP_val_mean, label = 'Validation')
      # plt.scatter(YP+YP_mean, RMSEC_predictions+YP_mean, label = 'Calibration')

      # ax.xaxis.set_major_locator(plt.MaxNLocator(7))
      # ax.grid()
      # ax.set_xlabel('Measured')
      # ax.set_ylabel('CV Predicted')
      # fig.suptitle(f'Actual vs. predicted {fileID} | KNN neighbors={neighbors}')
   
   plt.rcParams.update({'font.size': 14})

   # Flatten YP_val if it's a 2D array with one column
   YP_val = YP_val.flatten()

   fig, axes = plt.subplots(2, 2, figsize=(10, 10))
   axes = axes.flatten()
   fig.suptitle(f'Benchmarking Validation Scatter Plots | {fileID}', fontsize=16)

   models_predictions = {
      'PLS': M['pred']['val'][:,comps_p]+YP_mean,
      'Lasso': lasso_model['pred']['val'],
      'LightGBM': RMSEP_predictions,
      'KNN': RMSEP_predictions_knn
   }

   model_rmse = {
      'PLS': np.min(rmsep),
      'Lasso': np.min(rmsep_lasso),
      'LightGBM': rmsep_lgb,
      'KNN': rmsep_knn
   }


   for i, (model_name, model_predictions) in enumerate(models_predictions.items()):
      # Flatten model_predictions if it's not a 1D array
      if model_predictions.ndim > 1:
         model_predictions = model_predictions.flatten()

      # Calculate mean and standard deviation
      true_mean, true_std = np.mean(YP_val), np.std(YP_val)
      pred_mean, pred_std = np.mean(model_predictions), np.std(model_predictions)

      # Create a mask for filtering outliers
      mask = (YP_val > true_mean - 3*true_std) & (YP_val < true_mean + 3*true_std)
      mask = mask & (model_predictions > pred_mean - 3*pred_std) & (model_predictions < pred_mean + 3*pred_std)

      # Apply mask to filter outliers
      YP_val_filtered = YP_val[mask]
      cmap_filtered = cmap[mask]
      model_predictions_filtered = model_predictions[mask]
      # Plotting
      ax = axes[i]
      ax.scatter(YP_val_filtered, model_predictions_filtered, facecolors='none', edgecolors = cmap_filtered, alpha = 0.5)
      ax.set_title(f'{model_name} | RMSE = {round(float(model_rmse[model_name]), 3)} | n = {len(YP_val_filtered)}')
      ax.set_aspect('auto', 'box')
      ax.axline((0, 0), slope=1, color='k', linestyle='--')
      ax.set_xlabel('Measured')
      ax.set_ylabel('Predicted')

   plt.tight_layout()
   plt.savefig(f'figures/benchmarking/{fileID}_validation.png')
   plt.show()
   rmse_values = [round(float(value),3) for value in model_rmse.values()]
   with open('rmses.csv', 'a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(rmse_values) 





   # # Create 3 "super plots"
   # fig, super_axes = plt.subplots(1, 3, figsize=(15,5))
   # max_y_value = np.max(YP)*5 
   # print(max_y_value, np.max(YP))
   # # Loop through each "super plot" to create a 2x2 grid within it
   # for i, ax in enumerate(super_axes):
   #    # Create 2x2 inner gridspec


   #    inner_grid = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3,
   #                                  left=ax.get_position().x0, right=ax.get_position().x1, 
   #                                  bottom=ax.get_position().y0, top=ax.get_position().y1)
   #    # Create the 2x2 subplots
   #    ax1 = fig.add_subplot(inner_grid[0, 0])
   #    ax2 = fig.add_subplot(inner_grid[0, 1])
   #    ax3 = fig.add_subplot(inner_grid[1, 0])
   #    ax4 = fig.add_subplot(inner_grid[1, 1])

   #    all_handles = []
   #    all_labels = []

   #    # plot data in each subplot
   #    if i == 0:
   #       ax.set_title(f'Calibration | n={len(YP)}\n ')
   #       ax1.scatter(YP+YP_mean, M['pred']['cal'][:,comps_c]+YP_mean, facecolors='none', edgecolors='r', label = f'PLS | RMSEC : {np.min(rmsec)}')
   #       ax2.scatter(YP+YP_mean, lasso_model['pred']['cal']+YP_mean, facecolors='none', edgecolors='b', label = f'Lasso | RMSEC : {rmsec_lasso}')
   #       ax3.scatter(YP+YP_mean, RMSEC_predictions+YP_mean, facecolors='none', edgecolors='g', label = f'LGB | RMSEC : {rmsec_lgb}')
   #       ax4.scatter(YP+YP_mean, RMSEC_predictions_knn+YP_mean, facecolors='none', edgecolors='y', label = f'KNN | RMSEC : {rmsec_knn}')
   #    elif i == 1:
   #       ax.set_title(f'Cross-Validation | n={len(YP)}\n ')
   #       ax1.scatter(YP+YP_mean, M['pred']['cv'][:,comps_cv]+YP_mean, facecolors='none', edgecolors='r', label = f'PLS | RMSECV : {np.min(rmsecv)}')
   #       ax2.scatter(YP+YP_mean, lasso_model['pred']['cv']+YP_mean, facecolors='none', edgecolors='b', label = f'Lasso | RMSECV : {rmsecv_lasso}')
   #       ax3.scatter(YP+YP_mean, predictions+YP_mean, facecolors='none', edgecolors='g', label = f'LGB | RMSECV : {rmsecv_lgb}')
   #       ax4.scatter(YP+YP_mean, predictions_knn+YP_mean, facecolors='none', edgecolors='y', label = f'KNN | RMSECV : {rmsecv_knn}')  
   #    else:
   #       ax.set_title(f'Validation | n={len(YP_val)}\n ')
   #       ax1.scatter(YP_val+YP_mean, M['pred']['val'][:,comps_p]+YP_mean, facecolors='none', edgecolors='r', label = f'PLS | RMSEP : {np.min(rmsep)}')
   #       ax2.scatter(YP_val+YP_mean, lasso_model['pred']['val']+YP_mean, facecolors='none', edgecolors='b', label = f'Lasso | RMSEP : {rmsep_lasso} ')
   #       ax3.scatter(YP_val+YP_mean, RMSEP_predictions+YP_mean, facecolors='none', edgecolors='g', label = f'LGB | RMSEP : {rmsep_lgb}')
   #       ax4.scatter(YP_val+YP_mean, RMSEP_predictions_knn+YP_mean, facecolors='none', edgecolors='y', label = f'KNN | RMSEP : {rmsep_knn}')

   #    # Set axes to be equal and add axline to each subplot
   #    for ax_sub in [ax1, ax2, ax3, ax4]:
   #       ax_sub.set_aspect('auto', 'box')
   #       ax_sub.axline((0, 0), slope=1, color='k', linestyle='--')
   #       # ax_sub.set_xlim([0, max_y_value])
   #       # ax_sub.set_ylim([0, max_y_value])
   #       ax_sub.set_title(ax_sub.get_legend_handles_labels()[1][0], fontsize='x-small')
   #    # Remove ticks and labels from the "super plot"


   #    fig.suptitle(f'{fileID}', fontsize = 16)
   #    ax.set_xticks([])
   #    ax.set_yticks([])
   #    ax.spines['top'].set_visible(False)
   #    ax.spines['right'].set_visible(False)
   #    ax.spines['bottom'].set_visible(False)
   #    ax.spines['left'].set_visible(False)
   #    # ax.legend(all_handles, all_labels)
   # #plt.subplots_adjust(wspace=0,hspace=0)
   
   # plt.savefig(f'figures/benchmarking/{fileID}.png')