import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load dataset
data = pd.read_csv("logistics_data.csv")

print("Dataset loaded successfully!")
print(data)

# 2. Select features and target
X = data[["Shipment_Volume", "Transportation_Cost", "Distance"]]
y = data["Delivery_Time"]

# 3. Split data into training and testing sets
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Add constant column for Linear Regression
X_train_matrix = np.column_stack([
    np.ones(len(X_train)),
    X_train.values
])

# 5. Calculate regression coefficients
coefficients = np.linalg.lstsq(
    X_train_matrix,
    y_train.values,
    rcond=None
)[0]

intercept = coefficients[0]
feature_coefficients = coefficients[1:]

# 6. Make predictions
X_test_matrix = np.column_stack([
    np.ones(len(X_test)),
    X_test.values
])

y_pred = X_test_matrix @ coefficients

# 7. Calculate MAE
mae = np.mean(np.abs(y_test.values - y_pred))

# 8. Calculate RMSE
rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))

# 9. Calculate R-squared
ss_total = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
ss_residual = np.sum((y_test.values - y_pred) ** 2)

r2 = 1 - (ss_residual / ss_total)

print("\nModel Performance:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# 10. Display feature coefficients
print("\nFeature Coefficients:")

for feature, coefficient in zip(X.columns, feature_coefficients):
    print(feature, ":", round(coefficient, 4))

# 11. Create visualizations folder
os.makedirs("visualizations", exist_ok=True)

# 12. Actual vs Predicted graph
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Delivery Time")
plt.ylabel("Predicted Delivery Time")
plt.title("Actual vs Predicted Delivery Time")

plt.tight_layout()
plt.savefig("visualizations/actual_vs_predicted.png")
plt.show()

# 13. Feature Analysis graph
plt.figure(figsize=(8, 5))

plt.bar(X.columns, feature_coefficients)

plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Analysis for Delivery Time Prediction")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("visualizations/feature_analysis.png")
plt.show()

print("\nWeek 4 Predictive Modeling completed successfully!")