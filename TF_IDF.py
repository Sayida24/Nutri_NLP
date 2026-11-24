import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from preprocess import clean_ingredients

def prepare_features():
    # 1. Load the cleaned data from EDA
    # Note: Ensure EDA.py has been run to create this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'cleaned_ingredients.csv'))

    # 2. Unified Threshold Logic
    def apply_thresholds(cal_per_g):
        total_100g = cal_per_g * 100
        if total_100g <= 87.49: return 'Low'
        elif total_100g <= 260.87: return 'Medium'
        else: return 'High'

    df['label'] = df['cal/g'].apply(apply_thresholds)
    df['clean_text'] = df['ingr_name'].apply(clean_ingredients)

    # 3. TF-IDF Vectorization
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['clean_text'])
    
    # 4. Label Encoding (Low/Medium/High -> 0/1/2)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    # 5. Data Splitting (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Return the processed data and the tools needed for the web app
    return X_train, X_test, y_train, y_test, tfidf, le

if __name__ == "__main__":
    print("TF_IDF.py: Feature engineering logic is ready.")