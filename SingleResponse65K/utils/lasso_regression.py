import numpy as np
from sklearn import decomposition, metrics, model_selection
from sklearn.linear_model import Lasso

def lasso(cal_X, cal_y, val_X, cv, alpha_value):

    cv_pred = np.empty((cal_X.shape[0]), dtype=np.float64)
    if val_X is None:
        val_pred = None

    lasso_model = Lasso(alpha=alpha_value)

    rmse_values = []

    # Loop through cv splits
    for train_index, test_index in cv.split(cal_X):
        X_train, X_test = cal_X[train_index], cal_X[test_index]
        y_train, y_test = cal_y[train_index], cal_y[test_index]

        # Fit the model
        lasso_model.fit(X_train, y_train)

        # Predict
        cv_pred[test_index] = lasso_model.predict(X_test)
    
    
    # Fit on full calibration data and predict
    lasso_model.fit(cal_X, cal_y)
    cal_pred = lasso_model.predict(cal_X)

    # Optional: Predict on validation set
    if val_X is not None:
        val_pred = lasso_model.predict(val_X)    

    r = {'method':'regression','pred':{'cal':cal_pred, 'cv':cv_pred, 'val':val_pred}}
    return r
