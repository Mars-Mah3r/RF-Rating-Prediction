
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

file_path = "/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/5_Encoded.csv"
data = pd.read_csv(file_path)

X = data.drop(columns=['Rating']) 
y = data['Rating']  

# Remove classes with fewer than 2 samples
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
X = X[~y.isin(rare_classes)]
y = y[~y.isin(rare_classes)]

#  SMOTE to balance the dataset
smote = SMOTE(random_state=42, k_neighbors=1)
X_smote, y_smote = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote)

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
y_pred = best_rf_model.predict(X_test)

#  Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Feature Importance from Random Forest
feature_importances = best_rf_model.feature_importances_
features = X.columns

# dataFrame for feature importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
print("Top Features by Random Forest Importance:\n", importance_df)

plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top Features by Random Forest')
plt.gca().invert_yaxis()
plt.show()

explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

index_to_explain = 0 
shap.decision_plot(explainer.expected_value[1], shap_values[1][index_to_explain], X_test.iloc[index_to_explain])

# Permutation Importance
perm_importance = permutation_importance(best_rf_model, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({'Feature': features, 'Importance': perm_importance.importances_mean})
perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False).head(10)
print("Top Features by Permutation Importance:\n", perm_importance_df)

plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top Features by Permutation Importance')
plt.gca().invert_yaxis()
plt.show()
