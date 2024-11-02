# Credit Card Fraud Detection using Random Forest and XGBoost

This project aims to detect fraudulent transactions in a highly imbalanced credit card dataset using two powerful machine learning models: **Random Forest** and **XGBoost**. Through this project, we will demonstrate key steps in data preprocessing, model training, and hyperparameter tuning to achieve optimal results on imbalanced data.

## Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset** from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset contains anonymized credit card transactions, labeled as either fraudulent (1) or genuine (0).

- **Filename**: `creditcard.csv`
- **Size**: Approximately 284,807 transactions with 492 fraud cases.
- **Features**: The dataset includes time, amount, and 28 anonymized features (`V1`, `V2`, ..., `V28`) derived from PCA transformations.

## Project Overview

1. **Data Loading and Exploration**: Load the dataset, check for class imbalance, and visualize the distribution of fraud and non-fraud transactions.
2. **Data Preprocessing**: Normalize the `Amount` feature and split the data into training and testing sets.
3. **Model Training**: Train two models — Random Forest and XGBoost — with basic settings, followed by hyperparameter tuning using `GridSearchCV`.
4. **Evaluation**: Evaluate each model’s performance using precision, recall, F1-score, and a confusion matrix.

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/fraud-detection-project.git
    cd fraud-detection-project
    ```

2. **Install Dependencies**:
    Make sure to have Python installed, then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Script**:
    Place the `creditcard.csv` file in the project directory and run the main script:
    ```bash
    python main.py
    ```

## Explanation of `main.py`

The main script performs the following steps:

1. **Load Dataset**: Loads `creditcard.csv` and displays the first few rows for verification.
2. **Class Distribution Visualization**: Plots a bar graph (with a log-scale y-axis) to show the distribution between fraud and non-fraud classes.
3. **Feature Scaling**: Standardizes the `Amount` feature for easier model training.
4. **Data Splitting**: Splits the data into training and test sets.
5. **Random Forest and XGBoost Training**:
   - **Random Forest**: Initializes with basic parameters and performs training.
   - **XGBoost**: Initializes with basic parameters and performs training.
6. **Hyperparameter Tuning**: Uses `GridSearchCV` to optimize key hyperparameters for both models.
7. **Model Evaluation**: Evaluates each model on the test set, displaying classification reports and confusion matrices.

## Example Output

After running the script, you should see results including the best hyperparameters found for each model, as well as a performance report.

## Requirements

- Python
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib

## References

- [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud): Credit Card Fraud Detection
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

---

**Note**: Since `creditcard.csv` is a large dataset, it may be best to avoid uploading it to GitHub if not necessary.

---

### 3. **requirements.txt**

Generate the `requirements.txt` file to list all dependencies by running:

```bash
pip freeze > requirements.txt
