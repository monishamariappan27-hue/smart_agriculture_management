import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv(
    r"C:\Users\monis\OneDrive\Desktop\Smart_Agriculture_System\data\Crop_recommendation.csv"
)

# Features
X = df.drop("label", axis=1)

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df[["label", "Cluster"]].head())

print("\nCluster Counts:")
print(df["Cluster"].value_counts())

df.to_csv(
    r"C:\Users\monis\OneDrive\Desktop\Smart_Agriculture_System\data\clustered_farms.csv",
    index=False
)

print("Clustered dataset saved.")