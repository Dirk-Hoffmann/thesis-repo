from joblib import load
import pandas as pd
from visualize import main
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('data/allMoistureDirk.mat')
print(data)

spectral_data = data['A']
wavelengths = data['nm'][0]     

# Create DataFrame for X
X = pd.DataFrame(spectral_data, columns=wavelengths)

print(X)

response_data = data['Y']

# Create DataFrame for Y
Y = pd.DataFrame(response_data, columns=['Response'])  # Name the column as needed

response_dfs = {}

for col in Y.columns[:]:  # Skip the 'Abs2200' column
    response_dfs[col] = Y[[col]]
    print(col)
    print(response_dfs[col])
    main(X, response_dfs[col], str(col))
    plt.close('all')

