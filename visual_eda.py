import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    r"C:\Users\monis\OneDrive\Desktop\Smart_Agriculture_System\data\Crop_recommendation.csv"
)

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.drop("label", axis=1).corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Crop Distribution
plt.figure(figsize=(12,6))
sns.countplot(x="label", data=df)
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.show()