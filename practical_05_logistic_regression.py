import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
 
df = pd.read_csv("Social_Network_Ads.csv")
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
 
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print("Confusion matrix:\\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# Visualize the confusion matrix
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.colorbar()
plt.tight_layout()
plt.show()
