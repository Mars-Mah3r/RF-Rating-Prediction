import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

#Load  dataset
file_path = "/Users/mars/Documents/01-Applications/14-Moody's Rating/3rd Round/Data/5_Encoded.csv"
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

#Quick Hyperparameter Tuning with RandomizedSearchCV for LightGBM
param_dist = {
    'n_estimators': [100, 150, 200],           #  boosting rounds
    'max_depth': [10, 20, -1],                 # Maximum depth 
    'learning_rate': [0.05, 0.1, 0.2],         # Step size 
    'num_leaves': [31, 50, 70],                # Maximum no of leaves 
    'class_weight': ['balanced', None]         # Handling class imbalance
}

lgb_model = lgb.LGBMClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=10,                                 
    cv=3,                                      # Use 3-fold cross-validation
    n_jobs=-1,
    scoring='accuracy',
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_lgb_model = random_search.best_estimator_
y_pred = best_lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print("Best Parameters from RandomizedSearchCV:", random_search.best_params_)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
