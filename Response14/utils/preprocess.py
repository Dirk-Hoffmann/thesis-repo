import json
import hashlib
from threading import Thread, Event, Lock
import copy
import time
import psutil
from datetime import datetime
from joblib import Parallel, delayed
import progressbar

from hashlib import md5
import numpy as np
import numpy.matlib
from numpy.linalg import lstsq

import pandas as pd

from sklearn import cross_decomposition
from sklearn import decomposition
from sklearn import metrics
from sklearn import model_selection
from scipy import spatial
import sklearn
    
import time

def mean_center(X):
    f = np.mean(X, axis=0)
    X = X - f
    return (X, f)

def snv(X):
    X-= (X-np.mean(X)) / np.std(X)
    return X

def snv_inner(X, i):
    snv(X[i, :])

def snv_parallel(X, n_jobs = 8, verbose = 0):
    Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(delayed(snv_inner)(X, i) for i in range(X.shape[0]))

def detrend(X, wl=None, order=2):
    if wl is None:
        wl = np.arange(X.shape[-1])

    p = np.polyfit(wl, X, order)
    z = np.polyval(p, wl)
    X-= z
    return X

def detrend_inner(X, wl, i):   
    detrend(X[i,:], wl)

def detrend_parallel(X, wl=None, n_jobs = 8, verbose = 0):
    if wl is None:
        wl = range(X.shape[-1])
    Parallel(n_jobs=n_jobs, require="sharedmem", verbose=verbose)(delayed(detrend_inner)(X, wl, i) for i in range(X.shape[0]))

def msc(X, xref=None):
    if xref is None:
            xref = np.arange(0, X.shape[1])
    mean_vector = np.nanmean(X[:, xref], axis=0)
    A = np.vstack((np.ones(len(xref)), mean_vector))
    ab = lstsq(A.T, X[:, xref].T, rcond=None)[0]
    X = np.divide(np.subtract(X.T, ab[0, :]), ab[1, :]).T
    return (X, ab.T)