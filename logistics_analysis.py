import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv("amazon_delivery_data.csv")

# 2. Display basic information
print("First 5 records:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

# 3. Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 4. Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# 5. Basic statistics
print("\nBasic Statistics:")
print(data.describe())

# 6. Average delivery time
print("\nAverage Delivery Time:")
print(data["Delivery_Time"].mean())

# 7. Average delivery time by traffic condition
print("\nAverage Delivery Time by Traffic:")
print(data.groupby("Traffic")["Delivery_Time"].mean())

# 8. Average delivery time by weather
print("\nAverage Delivery Time by Weather:")
print(data.groupby("Weather")["Delivery_Time"].mean())

# 9. Average delivery time by vehicle
print("\nAverage Delivery Time by Vehicle:")
print(data.groupby("Vehicle")["Delivery_Time"].mean())

# 10. Simple visualization
data.groupby("Traffic")["Delivery_Time"].mean().plot(
    kind="bar",
    title="Average Delivery Time by Traffic Condition"
)

plt.xlabel("Traffic Condition")
plt.ylabel("Average Delivery Time")
plt.tight_layout()
plt.show()