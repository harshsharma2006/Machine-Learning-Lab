import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 
df = pd.read_csv("Advertising.csv")
# Expected columns: TV, Radio, Newspaper, Sales
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
 
comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": pred})
print(comparison.head(10).round(2))
print("MSE:", round(mean_squared_error(y_test, pred), 3))
print("R2 score:", round(r2_score(y_test, pred), 3))
 
comparison.head(10).plot(kind="bar", figsize=(8, 4))
plt.title("Actual vs Predicted Sales")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()
