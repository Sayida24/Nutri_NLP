import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. SETUP FOLDERS ---
# Create folders if they don't exist to avoid errors
if not os.path.exists('plots'): os.makedirs('plots')
if not os.path.exists('data/processed'): os.makedirs('data/processed')

# --- 2. LOAD DATA ---
try:
    df = pd.read_csv('data/ingredients_metadata.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'data/ingredients_metadata.csv' not found. Check your folder structure.")
    exit()

# --- 3. BASIC EXPLORATION ---
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicates ---")
print(f"Duplicates found: {df.duplicated().sum()}")

# --- 4. DATA CLEANING ---
df = df.drop_duplicates()
df = df.dropna()

# Remove invalid negative values
numeric_cols = ['cal/g', 'fat(g)', 'carb(g)', 'protein(g)']
for col in numeric_cols:
    df = df[df[col] >= 0]

# --- 5. VISUALIZATIONS & SAVING DIAGRAMS ---

# Diagram 1: Distribution Analysis (Univariate)
plt.figure(figsize=(10, 6))
sns.histplot(df['cal/g'], kde=True, color='teal')
plt.title('Calorie Density Distribution (cal/g)', fontsize=14)
plt.xlabel('Calories per Gram')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)
plt.savefig('plots/eda_1_distribution.png', dpi=300) # Save as high-res PNG
print("💾 Saved: plots/eda_1_distribution.png")


# Diagram 2: Outlier Analysis (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols], palette="Set2")
plt.title('Outlier Analysis for Nutrients', fontsize=14)
plt.ylabel('Value (g or cal/g)')
plt.savefig('plots/eda_2_outliers.png', dpi=300)
print("💾 Saved: plots/eda_2_outliers.png")

# Diagram 3: Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5)
plt.title('Nutrient Correlation Heatmap', fontsize=14)
plt.savefig('plots/eda_3_correlation.png', dpi=300)
print("💾 Saved: plots/eda_3_correlation.png")

# --- 6. SAVE CLEANED DATA ---
df.to_csv('data/processed/cleaned_ingredients.csv', index=False)
print("\n✅ EDA Complete!")
print("Cleaned data saved to: 'data/processed/cleaned_ingredients.csv'")

# Only show plots if running interactively
plt.show()