import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for visualizations
os.makedirs("visualizations", exist_ok=True)

# -----------------------------
# 1. Create Logistics Dataset
# -----------------------------

data = {
    "Shipment_ID": range(1, 21),

    "Delivery_Time": [
        2, 4, 3, 5, 6,
        3, 7, 4, 5, 2,
        6, 8, 3, 4, 7,
        5, 3, 6, 4, 5
    ],

    "Shipment_Volume": [
        50, 80, 60, 100, 120,
        70, 150, 90, 110, 45,
        130, 160, 65, 85, 140,
        105, 55, 125, 75, 95
    ],

    "Transportation_Cost": [
        500, 750, 600, 950, 1100,
        700, 1400, 850, 1000, 450,
        1200, 1500, 650, 800, 1350,
        980, 550, 1250, 720, 900
    ],

    "Distance": [
        20, 35, 25, 50, 60,
        30, 75, 40, 55, 15,
        65, 80, 28, 38, 70,
        52, 22, 68, 32, 45
    ]
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("logistics_data.csv", index=False)

print("========== LOGISTICS DATASET ==========")
print(df)


# -----------------------------
# 2. Exploratory Data Analysis
# -----------------------------

print("\n========== DATA INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== CENTRAL TENDENCY ==========")

print("Average Delivery Time:",
      df["Delivery_Time"].mean())

print("Median Delivery Time:",
      df["Delivery_Time"].median())

print("Average Shipment Volume:",
      df["Shipment_Volume"].mean())

print("Average Transportation Cost:",
      df["Transportation_Cost"].mean())

print("Average Distance:",
      df["Distance"].mean())


# -----------------------------
# 3. Correlation Analysis
# -----------------------------

correlation = df.corr(numeric_only=True)

print("\n========== CORRELATION MATRIX ==========")
print(correlation)


# -----------------------------
# 4. Visualization - Delivery Time
# -----------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["Delivery_Time"], bins=6)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")

plt.tight_layout()
plt.savefig("visualizations/delivery_time.png")
plt.show()


# -----------------------------
# 5. Visualization - Shipment Volume
# -----------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Shipment_ID"],
    df["Shipment_Volume"]
)

plt.title("Shipment Volume by Shipment ID")
plt.xlabel("Shipment ID")
plt.ylabel("Shipment Volume")

plt.tight_layout()
plt.savefig("visualizations/shipment_volume.png")
plt.show()


# -----------------------------
# 6. Visualization - Distance vs Cost
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Distance"],
    df["Transportation_Cost"]
)

plt.title("Distance vs Transportation Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost")

plt.tight_layout()
plt.savefig("visualizations/cost_distance.png")
plt.show()


# -----------------------------
# 7. Correlation Heatmap
# -----------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Between Logistics Variables")

plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png")
plt.show()


# -----------------------------
# 8. Final Message
# -----------------------------

print("\n======================================")
print("Analysis completed successfully!")
print("Dataset saved as logistics_data.csv")
print("All visualizations saved in visualizations folder.")
print("======================================")