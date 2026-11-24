import sys
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# 1. --- CONNECTION TO CENTRAL PIPELINE ---
# This allows Python to find TF_IDF.py in the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from TF_IDF import prepare_features
except ImportError:
    print("❌ ERROR: TF_IDF.py not found. Ensure it is in the main project folder.")
    sys.exit()

# 2. --- GET PROCESSED DATA FROM CENTRAL FILE ---
X_train, X_test, y_train, y_test, tfidf, le = prepare_features()

# 3. --- MODEL DEFINITION ---
model_name = "Random_Forest"
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. --- TRAIN AND PREDICT ---
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. --- PRINT PERFORMANCE METRICS ---
print(f"\n{'='*30}")
print(f"RESULTS FOR {model_name.replace('_', ' ')}")
print(f"{'='*30}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6. --- SAVE THE CONFUSION MATRIX AS AN IMAGE ---
if not os.path.exists('../plots'):
    os.makedirs('../plots')

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', # Purple map as requested
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix: {model_name}")
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

# Save to the root plots folder
plot_path = f"../plots/{model_name}_cm.png"
plt.savefig(plot_path)
print(f"\n✅ Success: Confusion Matrix saved to: {plot_path}")
plt.close()

