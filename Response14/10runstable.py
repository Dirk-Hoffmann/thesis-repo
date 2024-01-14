import pandas as pd

def calculate_average_rmse_per_reference(input_csv, output_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Assuming the first column is the reference
    reference_column = df.columns[0]

    # Group by the reference column and calculate the mean for each model
    average_rmse_per_reference = df.groupby(reference_column).mean()

    # Reset index to turn reference into a column again
    average_rmse_per_reference.reset_index(inplace=True)

    # Save the results to a new CSV file
    average_rmse_per_reference.to_csv(output_csv, index=False)

# Example usage
input_csv = '10runs.csv'
output_csv = '10runs_averaged_per_reference_no_dropout.csv'
calculate_average_rmse_per_reference(input_csv, output_csv)
