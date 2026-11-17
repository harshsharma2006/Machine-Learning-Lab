import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
 
# UCI HAR feature file and activity labels
X = pd.read_csv("X_train.txt", sep=r"\\s+", header=None)
y = pd.read_csv("y_train.txt", sep=r"\\s+", header=None).values.ravel()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
models = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "Accuracy:", round(accuracy_score(y_test, pred), 4))
 
# Activity IDs in the UCI HAR dataset: 1 to 6
print("Activity labels: 1=walking, 2=walking_upstairs, 3=walking_downstairs,")
print("4=sitting, 5=standing, 6=laying")
# Display activity distribution in the training labels
import matplotlib.pyplot as plt
pd.Series(y).value_counts().sort_index().plot(kind="bar", color="slateblue")
plt.title("Human Activity Class Distribution")
plt.xlabel("Activity ID")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.show()
