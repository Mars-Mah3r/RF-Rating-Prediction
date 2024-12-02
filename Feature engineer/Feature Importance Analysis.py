import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

file_path = '/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/5_Encoded.csv'
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop(columns=['Rating'])
y = data['Rating']

# Remove classes with fewer than 2 instances
y_class_counts = y.value_counts()
valid_classes = y_class_counts[y_class_counts >= 2].index
X_filtered = X[y.isin(valid_classes)]
y_filtered = y[y.isin(valid_classes)]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
feature_importances = rf_model.feature_importances_

# Create a DataFrame to rank features by importance
feature_importance_df = pd.DataFrame({
    'Feature': X_filtered.columns,
    'Importance': feature_importances
})

feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df_sorted['Cumulative Importance'] = feature_importance_df_sorted['Importance'].cumsum()
features_95 = feature_importance_df_sorted[feature_importance_df_sorted['Cumulative Importance'] <= 0.95]['Feature']

#  top 95% importance features 
X_filtered_95 = X_filtered[features_95]
filtered_data_95 = pd.concat([X_filtered_95, y_filtered.reset_index(drop=True)], axis=1)
filtered_data_95.to_csv('/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/6_Top_95_features.csv', index=False)

# Output the results for verification
print("Features covering 95% of importance:")
print(features_95)
