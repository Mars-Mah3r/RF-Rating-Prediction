import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb

file_path = "/Users/mars/Documents/01-Applications/14-Moodys Rating/3rd Round/Data/7_Final_Dataset.csv"
data = pd.read_csv(file_path)

X = data.drop(columns=['Rating']) 
y = data['Rating']  

class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
X = X[~y.isin(rare_classes)]
y = y[~y.isin(rare_classes)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_dist = {
    'n_estimators': [100, 150, 200],           
    'max_depth': [10, 20, -1],                 
    'learning_rate': [0.05, 0.1, 0.2],         
    'num_leaves': [31, 50, 70],               
    'class_weight': ['balanced', None]        
}

lgb_model = lgb.LGBMClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=10,                                
    cv=3,                                      
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

# Print results
print("Best Parameters from RandomizedSearchCV:", random_search.best_params_)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
