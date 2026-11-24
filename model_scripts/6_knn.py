import sys
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# 1. --- CONNECTION TO CENTRAL PIPELINE ---
# This allows Python to find TF_IDF.py in the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')

try:
    from TF_IDF import prepare_features
except ImportError:
    print("❌ ERROR: TF_IDF.py not found. Ensure it is in the main project folder.")
    sys.exit()

# 2. --- GET PROCESSED DATA FROM CENTRAL FILE ---
X_train, X_test, y_train, y_test, tfidf, le = prepare_features()

# 3. --- MODEL DEFINITION ---
model_name = "KNN"
# We use 5 neighbors as specified in your synopsis
model = KNeighborsClassifier(n_neighbors=5)

# 4. --- TRAIN AND PREDICT ---
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. --- PRINT PERFORMANCE METRICS ---
print(f"\n{'='*30}")
print(f"RESULTS FOR {model_name}")
print(f"{'='*30}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6. --- SAVE THE CONFUSION MATRIX AS AN IMAGE ---
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', # Yellow-Green-Blue map for KNN
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix: {model_name}")
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

# Save to the root plots folder
plot_path = os.path.join(PLOTS_DIR, f"{model_name}_cm.png")
plt.savefig(plot_path)
print(f"\n✅ Success: Confusion Matrix saved to: {plot_path}")
plt.close()

# 7. --- SAVE TOOLS (Final Model Export) ---
# Get the absolute path of the directory where this script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up)
BASE_DIR = os.path.dirname(CURRENT_DIR)
# Create the full path to the models folder
MODELS_PATH = os.path.join(BASE_DIR, 'models')

# Create the folder if it doesn't exist
if not os.path.exists(MODELS_PATH): 
    os.makedirs(MODELS_PATH)

# Save using the specific absolute path
pickle.dump(model, open(os.path.join(MODELS_PATH, 'best_model.pkl'), 'wb'))
pickle.dump(tfidf, open(os.path.join(MODELS_PATH, 'tfidf.pkl'), 'wb'))
pickle.dump(le, open(os.path.join(MODELS_PATH, 'encoder.pkl'), 'wb'))

print(f"\n🚀 SUCCESS: Model files exported to: {MODELS_PATH}")

