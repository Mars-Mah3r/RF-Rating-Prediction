import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE


file_path = "/Users/mars/Documents/01-Applications/14-Moody's Rating/3rd Round/Data/5_Encoded.csv"
data = pd.r
X = data.drop(columns=['Rating'])  
y = data['Rating']  

# Remove classes with fewer than 2 samples
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
X = X[~y.isin(rare_classes)]
y = y[~y.isin(rare_classes)]

smote = SMOTE(random_state=42, k_neighbors=1)
X_smote, y_smote = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote)

# GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],         
    'max_depth': [10, 20, 30, None],         
    'min_samples_split': [2, 5, 10],          
    'min_samples_leaf': [1, 2, 4],            
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': ['balanced', None]        
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print("Best Parameters from GridSearchCV:", grid_search.best_params_)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
