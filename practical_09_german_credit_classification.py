import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
 
df = pd.read_csv("german_credit_data.csv")
# Set TARGET to the actual target column in your downloaded dataset.
TARGET = "Risk"
X = df.drop(columns=[TARGET])
y = df[TARGET]
 
numeric = X.select_dtypes(include="number").columns
categorical = X.select_dtypes(exclude="number").columns
 
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred))
# Visualize predicted credit-risk classes
import matplotlib.pyplot as plt
pd.Series(pred).value_counts().plot(kind="bar", color=["seagreen", "indianred"])
plt.title("Predicted Credit Risk Distribution")
plt.xlabel("Risk class")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.show()
