from joblib import load
import pandas as pd
from visualize import main
import matplotlib.pyplot as plt
#from utils.datasplitter import X_train, y_train, X_test, X_val, y_test, y_val 
from utils.classes import SpectralDataset




# X = load('/home/djh/Big8_testing/data/big8/X.joblib')
# Y = load('/home/djh/Big8_testing/data/big8/Y.joblib')

# train_dataset = SpectralDataset(X_train, y_train, skip_normalization= True)
# test_dataset = SpectralDataset(X_test, y_test, skip_normalization= True)
# val_dataset = SpectralDataset(X_val, y_val, skip_normalization= True)


def benchmarking_models(train_dataset, test_dataset, val_dataset):
    df = pd.DataFrame(columns = ['PLS','Lasso','LightGBM','KNN'])

    response_dfs = {}

    for col in range(train_dataset.y.shape[1]):  # Skip the 'Abs2200' column
        response_dfs[col] = train_dataset.response_names[col]
        #print(response_dfs[col])
        #print(train_dataset.y[col])
    # print(response_dfs[col])
        benchmarking_rmses = main(train_dataset, test_dataset, val_dataset, train_dataset.response_names[col], col)
        
        benchmarking_df = pd.DataFrame(benchmarking_rmses, index = [train_dataset.response_names[col]])


        df = pd.concat([df, benchmarking_df], ignore_index=False)        
        plt.close('all')
    print(df)
    return df