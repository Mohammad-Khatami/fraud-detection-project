import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from pathlib import Path
import matplotlib.pyplot as plt

def plot_class_distribution(data):
    # Count the instances of each class
    class_counts = data['Class'].value_counts()
    
    # Plot the class distribution
    plt.figure(figsize=(6, 4))
    plt.bar(class_counts.index, class_counts.values, color=['blue', 'red'])
    plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'])
    plt.xlabel("Class")

    plt.yscale('log')  # Set y-axis to log scale

    plt.ylabel("Number of Transactions")
    plt.title("Class Distribution in Credit Card Fraud Dataset")
    plt.show()

# Load dataset
data = pd.read_csv("creditcard.csv")
print(data.head())

print(data['Class'].value_counts())


# Scaling data in "Amount" 
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
print(data['Amount'])

# prepare train and test datasets
X = data.drop(['Class'], axis=1)  # Features
y = data['Class']                 # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plot_class_distribution(data)

#--------------- Random Forest ------------------#
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],       # Number of trees
    'max_depth': [10, 15, 20],             # Maximum depth of trees
    'min_samples_split': [5, 10, 15],      # Minimum samples required to split
    'min_samples_leaf': [1, 5, 10],        # Minimum samples required in a leaf
    'max_features': ['sqrt', 'log2']       # Number of features considered per split
}

# Initialize the base Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Set up GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

# Retrieve and print the best parameters found
print("Best parameters found: ", grid_search.best_params_)

# Use the best parameters to initialize the final model
best_rf_model = grid_search.best_estimator_

# Train the final model on the training data
best_rf_model.fit(X_train, y_train)

### evaluate RF
rf_preds = best_rf_model.predict(X_test)
print("Optimized Random Forest Results:")
print(classification_report(y_test, rf_preds))
print(confusion_matrix(y_test, rf_preds))


#-------------------- XGBoost -------------------#
param_grid_xgb = {
    'n_estimators': [50, 100, 200],        # Number of boosting rounds (trees)
    'learning_rate': [0.01, 0.1, 0.2],     # Step size for updating weights (smaller values mean slower but more precise training)
    'max_depth': [3, 5, 7],                # Maximum depth of each tree
    'subsample': [0.6, 0.8, 1.0],          # Fraction of samples used per tree to prevent overfitting
    'colsample_bytree': [0.6, 0.8, 1.0],   # Fraction of features considered at each split to add randomness
    'gamma': [0, 0.1, 0.2]                 # Minimum loss reduction required to make a further partition (prevents overfitting)
}

# Initialize the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Set up GridSearchCV
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=3, scoring='f1')
grid_search_xgb.fit(X_train, y_train)

# Retrieve and print the best parameters found
print("Best parameters found for XGBoost: ", grid_search_xgb.best_params_)

# Use the best parameters to initialize the final model
best_xgb_model = grid_search_xgb.best_estimator_

# Train the final XGBoost model
best_xgb_model.fit(X_train, y_train)


### evaluate XGBoost
xgb_preds = best_xgb_model.predict(X_test)
print("Optimized XGBoost Results:")
print(classification_report(y_test, xgb_preds))
print(confusion_matrix(y_test, xgb_preds))


