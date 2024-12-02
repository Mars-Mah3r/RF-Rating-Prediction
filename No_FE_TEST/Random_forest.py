import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

file_path = "/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/5_Encoded.csv"
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=['Rating'])  # Features
y = data['Rating']  # Target

# Remove classes with fewer than 2 samples
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
X = X[~y.isin(rare_classes)]
y = y[~y.isin(rare_classes)]

smote = SMOTE(random_state=42, k_neighbors=1)
X_smote, y_smote = smote.fit_resample(X, y)

# Split the data into training, validation, test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42, stratify=y_smote)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


class_weights = dict()
total_samples = len(y_train)
unique_classes = y_train.unique()

for c in unique_classes:
    class_count = sum(y_train == c)
    class_weights[c] = total_samples / (len(unique_classes) * class_count)

print("Custom Class Weights:", class_weights)

#  Custom Class Weights
best_params = {
    'n_estimators': 300,           
    'max_depth': 30,                
    'min_samples_split': 2,         
    'min_samples_leaf': 1,          
    'max_features': 'sqrt',         
    'class_weight': class_weights,  
    'random_state': 42              
}

best_rf_model = RandomForestClassifier(**best_params)
best_rf_model.fit(X_train, y_train)

#ake predictions on the validation set
y_val_pred = best_rf_model.predict(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_classification_rep = classification_report(y_val, y_val_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)

#  validation results
print(f"Validation Accuracy: {val_accuracy}")
print("\nValidation Classification Report:\n", val_classification_rep)
print("\nValidation Confusion Matrix:\n", val_conf_matrix)
y_test_pred = best_rf_model.predict(X_test)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_classification_rep = classification_report(y_test, y_test_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy}")
print("\nTest Classification Report:\n", test_classification_rep)
print("\nTest Confusion Matrix:\n", test_conf_matrix)