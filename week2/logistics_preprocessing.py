import pandas as pd

# 1. Load the dataset
data = pd.read_csv("amazon_delivery_data.csv")

# 2. Display basic information
print("Dataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

# 3. Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 4. Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# 5. Check data types
print("\nData Types:")
print(data.dtypes)

# 6. Basic statistics
print("\nBasic Statistics:")
print(data.describe())

# 7. Check unique values in categorical columns
print("\nTraffic Values:")
print(data["Traffic"].unique())

print("\nWeather Values:")
print(data["Weather"].unique())

print("\nVehicle Values:")
print(data["Vehicle"].unique())


print("\n--- WEEK 2 PREPROCESSING RESULTS ---")

# Missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# Numerical columns
print("\nNumerical Columns:")
print(data.select_dtypes(include="number").columns.tolist())

# Categorical columns
print("\nCategorical Columns:")
print(data.select_dtypes(include="object").columns.tolist())

# Basic statistics
print("\nBasic Statistics:")
print(data.describe())

# Dataset shape
print("\nDataset Shape:")
print(data.shape)

# Unique values
print("\nTraffic Values:")
print(data["Traffic"].unique())

print("\nWeather Values:")
print(data["Weather"].unique())

print("\nVehicle Values:")
print(data["Vehicle"].unique())

# --- OUTLIER DETECTION ---

print("\n--- OUTLIER DETECTION ---")

numerical_columns = [
    "Agent_Age",
    "Agent_Rating",
    "Store_Latitude",
    "Store_Longitude",
    "Drop_Latitude",
    "Drop_Longitude",
    "Delivery_Time"
]

for column in numerical_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    print(column, ":", len(outliers), "outliers")

 # --- NORMALIZATION ---

print("\n--- NORMALIZATION ---")

normalized_data = data[numerical_columns].copy()

for column in numerical_columns:
    min_value = normalized_data[column].min()
    max_value = normalized_data[column].max()

    normalized_data[column] = (
        (normalized_data[column] - min_value)
        / (max_value - min_value)
    )

print("\nNormalized Data (First 5 Rows):")
print(normalized_data.head())

normalized_data.to_csv("preprocessed_logistics_data.csv", index=False)

print("\nPreprocessed dataset saved successfully.")

print("\n--- FINAL CLEANING CHECK ---")

print("Missing values after cleaning:")
print(data.isnull().sum().sum())

print("Duplicate records after cleaning:")
print(data.duplicated().sum())

print("Original dataset shape:")
print(data.shape)

print("Normalized dataset shape:")
print(normalized_data.shape)