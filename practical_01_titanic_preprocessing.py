import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
 
# Load the Titanic CSV file
df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset shape:", df.shape)
print("\\nMissing values before preprocessing:")
print(df.isnull().sum())
 
# Select useful numeric columns for this demonstration
features = [c for c in ["Pclass", "Age", "SibSp", "Parch", "Fare"] if c in df.columns]
X = df[features].copy()
y = df["Survived"]
 
# Replace missing numeric values with the median
imputer = SimpleImputer(strategy="median")
X[features] = imputer.fit_transform(X[features])
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
print("\\nMissing values after preprocessing:", int(X.isnull().sum().sum()))
print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)
# Visualize missing values before preprocessing
import matplotlib.pyplot as plt
missing = df.isnull().sum().sort_values(ascending=False).head(10)
missing = missing[missing > 0]
if not missing.empty:
    missing.plot(kind="bar", color="steelblue", figsize=(8, 4))
    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing count")
    plt.tight_layout()
    plt.show()
