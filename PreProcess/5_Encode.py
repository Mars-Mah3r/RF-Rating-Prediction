import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the specified file path
file_path = '/Users/mars/Downloads/4_Correlation_Corrected.csv'
data = pd.read_csv(file_path)

# Create a copy to apply transformations
encoded_data = data.copy()

# 1. Label Encoding for "Rating"
label_encoder = LabelEncoder()
encoded_data['Rating'] = label_encoder.fit_transform(encoded_data['Rating'])

# 2. Frequency Encoding for "Name"
name_freq = encoded_data['Name'].value_counts(normalize=True)
encoded_data['Name'] = encoded_data['Name'].map(name_freq)

# 3. One-Hot Encoding for "Rating Agency Name"
encoded_data = pd.get_dummies(encoded_data, columns=['Rating Agency Name'], prefix='Agency', drop_first=True)

# 4. Converting "Date" to Year, Month, and Day
encoded_data['Date'] = pd.to_datetime(encoded_data['Date'], errors='coerce')
encoded_data['Year'] = encoded_data['Date'].dt.year
encoded_data['Month'] = encoded_data['Date'].dt.month
encoded_data['Day'] = encoded_data['Date'].dt.day
encoded_data.drop(columns=['Date'], inplace=True)

# 5. One-Hot Encoding for "Sector"
encoded_data = pd.get_dummies(encoded_data, columns=['Sector'], prefix='Sector', drop_first=True)

# Save the encoded dataset to a new file if needed
output_file_path = '/Users/mars/Downloads/5_Encoded.csv'
encoded_data.to_csv(output_file_path, index=False)

print(f"Encoded data has been saved to: {output_file_path}")
