from utils.datasplitter import splitter
from CNN import run_CNN
from SCNN_batchfixed import run_SCNN
from plotShallowCNN import plot_SCNN
from plots_CNN import plot_CNN
from benchmarking_models import benchmarking_models
import pandas as pd

CNN_rmse_output = {}
benchmarking_rmse_output = {}

df = pd.DataFrame(columns = ['PLS', 'Lasso', 'LightGBM', 'KNN', 'SCNN','DCNN'])

print(df)

for i in range(10):
    cmap, train_dataset, test_dataset, val_dataset = splitter(RANDOM_STATE=i, skip_normalization=True)
    benchmarking_df = benchmarking_models(train_dataset, test_dataset, val_dataset)

    cmap, train_dataset, test_dataset, val_dataset = splitter(RANDOM_STATE=i)
    val_loader = run_CNN(train_dataset, val_dataset, test_dataset, name= str(i))
    CNN_df = plot_CNN(val_loader, name = str(i))

    val_loader = run_SCNN(train_dataset, val_dataset, test_dataset, name= str(i))
    SCNN_df = plot_SCNN(val_loader, name = str(i))

    loop_df = pd.concat([benchmarking_df,CNN_df,SCNN_df], axis=1)

    df = pd.concat([df, loop_df], axis=0)
 


df.to_csv('10runs.csv')
print(df)

