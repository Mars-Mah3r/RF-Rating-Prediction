import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

file_path = "/Users/mars/Documents/Top_90_Percent_Features_Engineered_Dataset.csv"
data = pd.read_csv(file_path)

X = data.drop(columns=['Rating']) 
y = data['Rating'] 
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 3].index
X = X[~y.isin(rare_classes)]
y = y[~y.isin(rare_classes)]

smote = SMOTE(sampling_strategy="auto", k_neighbors=1, random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
feature_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

important_features = feature_importance_df[feature_importance_df['Importance'] > 0]['Feature']
X_train_filtered = X_train[important_features]
X_test_filtered = X_test[important_features]

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [{5: 5, **{cls: 1 for cls in y_res.unique() if cls != 5}}, 'balanced']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)
grid_search.fit(X_train_filtered, y_train)
best_rf_model = grid_search.best_estimator_

# Evaluate the Model
y_pred = best_rf_model.predict(X_test_filtered)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Save Feature Importance for Reference
feature_importance_df.to_csv("/Users/mars/Documents/Feature_Importance.csv", index=False)

# Print Important Features for Verification
print("Important Features:\n", important_features)
