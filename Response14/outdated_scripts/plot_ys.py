import matplotlib.pyplot as plt
from copy import copy
import numpy as np
import pandas as pd

fileID = 'protein'

Y_train = pd.read_csv(f'data/y_{fileID}_300.csv').set_index('sampleid')

YP = copy(Y_train.to_numpy())

std_YP = np.std(YP, axis=0)

# Create filters based only on Y-values
filter_YP = np.all(abs(YP) < 5 * std_YP, axis=1)

YP = YP[filter_YP]


# Plot histogram
plt.hist(YP, bins=50, density=True)
plt.xlabel("Y-values")
plt.ylabel("Frequency")
plt.title("Distribution of Y-values")

plt.show()
plt.savefig(f'figures/{fileID}_y-values.png')

