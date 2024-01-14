import json
import re
from datetime import datetime
from hashlib import md5
from pandas import Series

import bson
import numpy as np
from multiprocessing import Value, Lock

class BimseEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bson.ObjectId):
            return str(obj)
        if isinstance(obj, list):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, Series):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)
    
def array_hash(a):
    return md5(a[:]).hexdigest()
    
def rms(a):
    return np.sqrt(np.mean(a[:]**2))

def rmse(a, b):
    aa = np.squeeze(a) 
    bb = np.squeeze(b)
    if aa.ndim == 1:
        aa = aa.reshape(-1,1)
    if bb.ndim == 1:
        bb = bb.reshape(-1,1)
    return np.sqrt(np.mean(np.subtract(aa, bb)**2, axis=0))


def corr(a, b):
    aa = np.squeeze(a) 
    bb = np.squeeze(b)
  
    if aa.ndim == 1:
        aa = aa.reshape(-1,1)
    if bb.ndim == 1:
        bb = bb.reshape(-1,1)
        
    if aa.shape[0] != bb.shape[0]:
        return None
    
    if aa.shape[1] == 1 and bb.shape[1] == 1:
        return np.corrcoef(aa,bb)
    
    elif aa.shape[1] == 1 and bb.shape[1] > 1:
        out = np.full((bb.shape[1],), np.nan, np.float64)
        for n in range(bb.shape[1]):
            out[n] = np.corrcoef(aa.squeeze(),bb[:,n].squeeze())[0,1]
        return out

    elif aa.shape[1] > 1 and bb.shape[1] == 1:
        out = np.full((aa.shape[1],), np.nan, np.float64)
        for n in range(aa.shape[1]):
            out[n] = np.corrcoef(bb.squeeze(),aa[:,n].squeeze())[0,1]
        return out

    else:
        return None

def bias(a, b):
    aa = np.squeeze(a) 
    bb = np.squeeze(b)
    
    if aa.ndim == 1:
        aa = aa.reshape(-1,1)
    if bb.ndim == 1:
        bb = bb.reshape(-1,1)
    return np.mean(np.subtract(aa, bb), axis=0)


def sep(a, b):
    aa = np.squeeze(a) 
    bb = np.squeeze(b)
    if aa.ndim == 1:
        aa = aa.reshape(-1,1)
    if bb.ndim == 1:
        bb = bb.reshape(-1,1)
    return np.sqrt((np.sum((np.subtract(aa, bb)-bias(a,b))**2, axis=0))/(a.shape[0]-1))

def optlv(rmsecv, default_comp = 0, change_lim = 5):
    # Find number of components based on CV rmse
    # 1. check for first local error minimum 
    for n in range(1, len(rmsecv)-1):
        if rmsecv[n-1] < rmsecv[n]:
            return n-1

    # 2. If no local minimum, find first shoulder (e.g. < n% rmsecv change between components)
    for n in range(1, len(rmsecv)-1):
        if np.abs(100-100*rmsecv[n-1]/rmsecv[n]) < change_lim:
            return n-1
  
    return default_comp

def snake2camelBack(name: str) -> str:
    return re.sub(r'_([a-z])', lambda x: x.group(1).upper(), name)

def snake2camel(name: str) -> str:
    return re.sub(r'(?:^|_)(\w)', lambda x: x.group(1).upper(), name)

def camel2snake(name: str) -> str:
    return name[0].lower() + re.sub(r'(?!^)[A-Z]', lambda x: '_' + x.group(0).lower(), name[1:])

def camelBack2snake(name: str) -> str:
    return re.sub(r'[A-Z]', lambda x: '_' + x.group(0).lower(), name)

