import pandas as pd

# Load dataset
df = pd.read_csv(
    r"C:\Users\monis\OneDrive\Desktop\Smart_Agriculture_System\data\Crop_recommendation.csv"
)

print("\n===== Dataset Shape =====")
print(df.shape)

print("\n===== Column Names =====")
print(df.columns)

print("\n===== Dataset Information =====")
print(df.info())

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Statistical Summary =====")
print(df.describe())

print("\n===== Crop Labels =====")
print(df['label'].unique())

print("\n===== Number of Crop Types =====")
print(df['label'].nunique())

print("\n===== Crop Counts =====")
print(df['label'].value_counts())