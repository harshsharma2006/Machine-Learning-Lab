import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
df = pd.read_csv("HousePrices.csv")
# Expected columns: Area, Bedrooms, Age, Price
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
 
print("Actual prices:", list(y_test.head()))
print("Predicted prices:", pred[:5].round(2))
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 2))
print("R2 score:", round(r2_score(y_test, pred), 3))
# Visualize actual and predicted house prices
plt.figure(figsize=(7, 4))
plt.scatter(y_test, pred, alpha=0.75, color="teal")
limits = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(limits, limits, "r--", label="Ideal prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()
