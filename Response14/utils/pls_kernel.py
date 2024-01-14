#%%
import numpy as np
import numpy.linalg as la
from threadpoolctl import threadpool_limits
from utils.RealWildStuff.improved_kernel_pls import *
    
def pls(cal_X, cal_y, val_X, n_components, cv, thread_limit=1, verbosity=1):

    cv_pred = np.empty((cal_X.shape[0], n_components), dtype=np.float64)
    if val_X is None:
        val_pred = None

    # with threadpool_limits(limits=thread_limit, user_api='blas'):

    # val pred
    P = PLS(algorithm=1, print_stuff=True if verbosity > 2 else False,  print_important_stuff=True if verbosity>0 else False )

    P.fit(cal_X, cal_y, n_components)
    
    cal_pred = P.predict(cal_X).T
    if val_X is not None: 
        val_pred = P.predict(val_X).T
    else:
        val_pred = None
    
    # cv pred
    for c_ix, v_ix in cv.split(X=cal_X, y=cal_y):
        P.fit(cal_X[c_ix,:], cal_y[c_ix], n_components)
        cv_pred[v_ix, :] = P.predict(cal_X[v_ix,:]).T

    r = { 'method': 'kernel', 'pred' : { 'cal': cal_pred, 'cv': cv_pred, 'val': val_pred }, 'n_components': n_components}
    return r


if __name__ == '__main__':

    from sklearn import model_selection
    from threadpoolctl import threadpool_limits
    import matplotlib.pyplot as plt
    from numpy import matlib as ml
    import time
    from joblib import Parallel, delayed

    cal_X = np.random.uniform(low=0, high=1, size=(100, 4200)).astype(np.float64)
    cal_y = np.random.uniform(low=0, high=1, size=(100, 1)).astype(np.float64)
    val_X = np.random.uniform(low=0, high=1, size=(1, 4200)).astype(np.float64)
    n_components = 20
    cv = model_selection.LeaveOneOut()

    starttime = time.time()
    #for c_ix, v_ix in cv.split(X=cal_X):
    #    cv_r_pred.append(pls_pred_inner(cal_X, cal_y, c_ix, v_ix))

    r = pls(cal_X, cal_y, val_X, 20, cv, thread_limit=1)

    elapsed = time.time() - starttime
    print(f"Elapsed: {elapsed:.3f}s")

    
