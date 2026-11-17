# Machine Learning Lab Practicals

Python code for 11 Machine Learning laboratory practicals.

## Practical list
1. Titanic dataset preprocessing and train/test split
2. Linear Regression – house price prediction
3. Multiple Linear Regression – advertising sales prediction
4. PCA – Iris dataset
5. Logistic Regression – Social Network Ads
6. K-Nearest Neighbors – diabetes classification
7. SVM and Naive Bayes – spam classification
8. Decision Tree – breast cancer classification
9. German Credit risk classification
10. K-Means and Hierarchical clustering – Mall Customers
11. Human Activity Recognition classification

## Setup
```bash
python -m pip install -r requirements.txt
```

Run any practical, for example:
```bash
python practical_01_titanic_preprocessing.py
```

## Dataset files
Put the required datasets in this folder before running the related practical:
- `Titanic-Dataset.csv`
- `HousePrices.csv` (columns: `Area`, `Bedrooms`, `Age`, `Price`)
- `Advertising.csv` (columns: `TV`, `Radio`, `Newspaper`, `Sales`)
- `Social_Network_Ads.csv` (columns: `Age`, `EstimatedSalary`, `Purchased`)
- `diabetes.csv` (target column: `Outcome`)
- `emails.csv` (columns: `text`, `label`; the spam class should be labelled `spam`)
- `german_credit_data.csv` (target column: `Risk`; change `TARGET` in the script if needed)
- `Mall_Customers.csv`
- UCI HAR files: `X_train.txt` and `y_train.txt`

The Iris and Breast Cancer practicals use datasets included with scikit-learn.

**Note:** Dataset files themselves are not included. Check dataset column names and label formats against each script before running it.
