import pandas as pd
import numpy as np
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

file_path = '/Users/mars/Downloads/2_Filtered_data.csv'
data = pd.read_csv(file_path)

#Copy data for transformations
transformed_data = data.copy()

# Identify numeric columns
numeric_columns = transformed_data.select_dtypes(include=['float64', 'int64']).columns

# Identify columns with high skewness
skewed_columns = transformed_data[numeric_columns].skew()
highly_skewed_columns = skewed_columns[skewed_columns.abs() > 1].index

# Apply log transformation with shift for non-positive values
for column in highly_skewed_columns:
    if (transformed_data[column] <= 0).any():
        shift_value = abs(transformed_data[column].min()) + 1
        transformed_data[column] = np.log(transformed_data[column] + shift_value)
    else:
        transformed_data[column] = np.log(transformed_data[column])

# Visualise distributions before and after log transformation
for column in highly_skewed_columns:
    plt.figure(figsize=(12, 5))

    # Original distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f"Original Distribution of {column}")
    plt.xlabel(column)

    # Log-transformed distribution
    plt.subplot(1, 2, 2)
    sns.histplot(transformed_data[column], kde=True, bins=30)
    plt.title(f"Log-Transformed Distribution of {column}")
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()

#Box-Cox Transformation
# Apply Box-Cox transformation to specific columns 
columns_to_boxcox = ['returnOnEquity', 'netProfitMargin']
for column in columns_to_boxcox:
    if (transformed_data[column] > 0).all():
        transformed_data[column], _ = boxcox(transformed_data[column])
    else:
        # Shift data to handle non-positive 
        shift_value = abs(transformed_data[column].min()) + 1
        transformed_data[column], _ = boxcox(transformed_data[column] + shift_value)

# Z-Score Normalization
scaler = StandardScaler()
transformed_data[numeric_columns] = scaler.fit_transform(transformed_data[numeric_columns])


output_file_path = '/Users/mars/Downloads/3_Normalized_Data.csv'
transformed_data.to_csv(output_file_path, index=False)
print(f"Transformed dataset saved to: {output_file_path}")
