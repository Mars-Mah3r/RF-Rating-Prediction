import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/Users/mars/Desktop/3_Normalized_Data.csv'
data = pd.read_csv(file_path)

numeric_data = data.select_dtypes(include=['float64', 'int64'])

# correlation matrix for numeric data
plt.figure(figsize=(12, 10))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix (Before Dropping Features)")
plt.show()

# Drop redundant numeric features based on correlation
features_to_drop = [
    'pretaxProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
    'ebitPerRevenue', 'companyEquityMultiplier', 'quickRatio'
]
reduced_numeric_data = numeric_data.drop(columns=features_to_drop, errors='ignore')

# Combine reduced numeric data with original string columns
non_numeric_data = data.select_dtypes(exclude=['float64', 'int64'])
final_data = pd.concat([non_numeric_data, reduced_numeric_data], axis=1)

# Step 3: Recompute the correlation matrix for reduced numeric data
plt.figure(figsize=(10, 8))
reduced_correlation_matrix = reduced_numeric_data.corr()
sns.heatmap(reduced_correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix (After Dropping Features)")
plt.show()

output_file_path = '/Users/mars/Documents/01-Applications/14-Moody\'s\ Rating/3rd\ Round/Data/4_Correlation_Corrected.csv'
final_data.to_csv(output_file_path, index=False)
print(f"Reduced dataset saved to: {output_file_path}")
