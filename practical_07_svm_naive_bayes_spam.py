import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
 
df = pd.read_csv("emails.csv")
# Expected columns: text and label (spam/ham or 1/0)
X = df["text"].fillna("")
y = df["label"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
 
models = {
    "SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB()
}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    print("\\n", name)
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred, average="binary", pos_label="spam"), 4))
    print("Recall   :", round(recall_score(y_test, pred, average="binary", pos_label="spam"), 4))
    print("F1-score :", round(f1_score(y_test, pred, average="binary", pos_label="spam"), 4))
# Compare model performance visually
import matplotlib.pyplot as plt
metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
svm_values, nb_values = [], []
for estimator, target in [(LinearSVC(), svm_values), (MultinomialNB(), nb_values)]:
    estimator.fit(X_train_vec, y_train)
    pred_plot = estimator.predict(X_test_vec)
    target.extend([
        accuracy_score(y_test, pred_plot),
        precision_score(y_test, pred_plot, average="binary", pos_label="spam"),
        recall_score(y_test, pred_plot, average="binary", pos_label="spam"),
        f1_score(y_test, pred_plot, average="binary", pos_label="spam")
    ])
positions = range(len(metric_names))
plt.figure(figsize=(8, 4))
plt.bar([v - 0.18 for v in positions], svm_values, width=0.36, label="SVM")
plt.bar([v + 0.18 for v in positions], nb_values, width=0.36, label="Naive Bayes")
plt.xticks(list(positions), metric_names)
plt.ylim(0, 1.05)
plt.title("SVM vs Naive Bayes Performance")
plt.legend()
plt.tight_layout()
plt.show()
