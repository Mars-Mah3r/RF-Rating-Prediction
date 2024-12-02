import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/0_Original_Dataset.csv'
data = pd.read_csv(file_path)

# Remove duplicate rows
data = data.drop_duplicates()
print(f"Remaining rows after removing duplicates: {data.shape[0]}")

# Calculate the percentage of missing values for each column
missing_percentage = data.isnull().mean() * 100

# Drop columns with more than 50% missing values
columns_to_drop = missing_percentage[missing_percentage > 50].index
data = data.drop(columns=columns_to_drop)
print(f"Columns dropped due to excessive missing values: {columns_to_drop.tolist()}")

columns_to_check = ['returnOnEquity', 'companyEquityMultiplier']

print(data.info())

for column in columns_to_check:
    # Calculate skewness
    skewness = data[column].skew()
    print(f"Skewness for {column}: {skewness}")
    
    # Plot histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(data[column].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

    # Plot boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[column])
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.show()

    # Determine imputation strategy
    if abs(skewness) < 0.5:
        print(f"{column} is approximately normally distributed. Recommend using MEAN for imputation.\n")
    else:
        print(f"{column} is skewed. Recommend using MEDIAN for imputation.\n")


columns_to_impute = ['returnOnEquity', 'companyEquityMultiplier']

for column in columns_to_impute:
    # Calculate the median of the column (ignoring NaN values)
    median_value = data[column].median()
    print(f"Median value for {column}: {median_value}")
    
    # Fill missing values with the median
    data[column].fillna(median_value, inplace=True)

# Verify no missing values remain in the specified columns
missing_values = data[columns_to_impute].isnull().sum()
print("Missing values after imputation:\n", missing_values)

# Save the processed DataFrame to a CSV file
output_file_path = '/Users/mars/Downloads/1_NaN_removed_data.csv' 
data.to_csv(output_file_path, index=False) 

print(f"Processed dataset has been exported to: {output_file_path}")
