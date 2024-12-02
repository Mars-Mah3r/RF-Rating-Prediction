import pandas as pd
import numpy as np

# Load the dataset
file_path = '/Users/mars/Downloads/1_NaN_removed_data.csv'
data = pd.read_csv(file_path)

#Filter out rows with negative values in specific columns
columns_to_filter = [
    'currentRatio', 'quickRatio', 'cashRatio',
    'daysOfSalesOutstanding', 'assetTurnover',
    'fixedAssetTurnover', 'payablesTurnover'
]

# Remove rows with negative values in the specified columns
filtered_data = data[~(data[columns_to_filter] < 0).any(axis=1)]
print(f"Original dataset size: {data.shape[0]}")
print(f"Filtered dataset size: {filtered_data.shape[0]}")

#Identify all numeric columns for winsorization
numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64']).columns

# Apply winsorization (1st and 99th percentiles)
for column in numeric_columns:
    lower_bound = filtered_data[column].quantile(0.01)
    upper_bound = filtered_data[column].quantile(0.99)
    filtered_data[column] = np.where(
        filtered_data[column] < lower_bound, lower_bound,
        np.where(filtered_data[column] > upper_bound, upper_bound, filtered_data[column])
    )

output_file_path = '/Users/mars/Downloads/2_Filtered_data.csv'
filtered_data.to_csv(output_file_path, index=False)

print(f"Processed dataset saved to: {output_file_path}")
