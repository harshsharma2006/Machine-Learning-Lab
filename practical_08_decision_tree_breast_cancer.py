from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
 
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
 
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=data.target_names))
 
new_sample = X_test[0].reshape(1, -1)
prediction = model.predict(new_sample)[0]
print("New sample classification:", data.target_names[prediction])
 
plt.figure(figsize=(14, 7))
plot_tree(model, feature_names=data.feature_names,
          class_names=data.target_names, filled=True, max_depth=3)
plt.tight_layout()
plt.show()
