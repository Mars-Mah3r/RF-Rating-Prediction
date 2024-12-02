import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

file_path = "/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/5_Encoded.csv"
data = pd.read_csv(file_path)
X = data.drop(columns=['Rating']) 
y = data['Rating'] 
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
X = X[~y.isin(rare_classes)]
y = y[~y.isin(rare_classes)]

smote = SMOTE(random_state=42, k_neighbors=1)
X_smote, y_smote = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote)

#Calculate Custom Class Weights
class_weights = dict()
total_samples = len(y_train)
unique_classes = y_train.unique()

for c in unique_classes:
    class_count = sum(y_train == c)
    class_weights[c] = total_samples / (len(unique_classes) * class_count)

print("Custom Class Weights:", class_weights)

best_params = {
    'n_estimators': 300,            
    'max_depth': 30,                
    'min_samples_split': 2,        
    'min_samples_leaf': 1,        
    'max_features': 'auto',        
    'class_weight': class_weights,  
    'random_state': 42             
}


best_rf_model = RandomForestClassifier(**best_params)
best_rf_model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = best_rf_model.predict(X_test)
##Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
