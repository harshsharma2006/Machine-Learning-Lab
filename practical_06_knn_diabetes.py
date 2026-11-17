import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
 
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
scores = {}
for k in range(1, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores[k] = accuracy_score(y_test, knn.predict(X_test))
 
for k, score in scores.items():
    print(f"K={k}: Accuracy={score:.3f}")
best_k = max(scores, key=scores.get)
print("Best K:", best_k)
 
plt.plot(list(scores.keys()), list(scores.values()), marker="o")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN accuracy for different K values")
plt.grid(True)
plt.show()
