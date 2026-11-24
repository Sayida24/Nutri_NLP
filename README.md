# 🥗 NutriNLP: Intelligent Food Prediction

NutriNLP is a Streamlit web app for nutrition analysis. You type a meal in plain English, such as `200g pasta and 50g butter`, and it:

- finds each ingredient in a database of about 550 foods, using fuzzy matching so small spelling mistakes still work;
- calculates the meal's total **calories, protein, fat and carbs**;
- rates the meal as **Low**, **Medium** or **High** in calories;
- suggests **healthier swaps**, for example olive oil or Greek yogurt in place of butter;
- shows how a trained **machine learning classifier** (TF-IDF + KNN) labels the meal text.

This is a course project for **SE333: Artificial Intelligence**.

## How it works

1. **Data cleaning** (`eda.py`): loads `data/ingredients_metadata.csv`, removes duplicates, missing values and negative values, saves the clean copy to `data/processed/cleaned_ingredients.csv`, and draws the EDA charts.
2. **Text preprocessing** (`preprocess.py`): lowercases the text, strips punctuation, removes stopwords and lemmatizes words with NLTK.
3. **Feature engineering** (`TF_IDF.py`): gives each ingredient a calorie label based on calories per 100 g, turns ingredient names into TF-IDF vectors, and splits the data 80/20 into training and test sets.

   | Label | Calories per 100 g |
   |-------|--------------------|
   | Low | up to 87.49 |
   | Medium | 87.50 to 260.87 |
   | High | above 260.87 |

4. **Model training** (`model_scripts/`): eight classifiers are trained and compared, and each one saves a confusion matrix to `plots/`.
5. **Web app** (`app.py`): the Streamlit interface. It loads the saved model from `models/`.

## Models compared

| # | Script | Model |
|---|--------|-------|
| 1 | `1_logistic_regression.py` | Logistic Regression |
| 2 | `2_multinomial_nb.py` | Multinomial Naive Bayes |
| 3 | `3_decision_tree.py` | Decision Tree |
| 4 | `4_random_forest.py` | Random Forest |
| 5 | `5_svm.py` | Support Vector Machine |
| 6 | `6_knn.py` | K-Nearest Neighbors (k = 5) |
| 7 | `7_gradient_boosting.py` | Gradient Boosting |
| 8 | `8_xgboost.py` | XGBoost |

**KNN** is the model the app uses. `6_knn.py` is the script that exports `best_model.pkl`, `tfidf.pkl` and `encoder.pkl` to `models/`.

## Project structure

```
Nutri_NLP/
├── app.py                  # Streamlit web app
├── preprocess.py           # NLP text cleaning
├── TF_IDF.py               # Labels, TF-IDF features, train/test split
├── eda.py                  # Data cleaning and EDA charts
├── requirements.txt
├── data/
│   ├── ingredients_metadata.csv          # Raw ingredient nutrition data
│   └── processed/cleaned_ingredients.csv # Cleaned data
├── model_scripts/          # One training script per model
├── models/                 # Saved model, TF-IDF vectorizer, label encoder
├── plots/                  # EDA charts and confusion matrices
└── templates/index.html    # Earlier Flask version of the interface
```

## Getting started

You need **Python 3.10 or newer**.

```bash
# 1. Clone the repository
git clone https://github.com/Sayida24/Nutri_NLP.git
cd Nutri_NLP

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

Streamlit then opens the app in your browser, usually at http://localhost:8501. The trained model is already in `models/`, so you don't need to retrain anything before running the app.

On first run, NLTK downloads its `stopwords` and `wordnet` data automatically.

## Retraining (optional)

You can run these from any folder:

```bash
python eda.py                     # clean the data and redraw the EDA charts
python model_scripts/6_knn.py     # retrain KNN and re-export the model files
```

Each of the other scripts in `model_scripts/` trains one of the comparison models and prints its accuracy and classification report.

## Example

**Input:** `100g white rice, 50g butter`

**Output:** the total calories and macronutrients, a per-ingredient calorie table, the meal's calorie category, and swaps such as *Brown Rice, Quinoa* for rice and *Olive Oil, Greek Yogurt* for butter.

## Tech stack

Python · Streamlit · scikit-learn · XGBoost · NLTK · pandas · NumPy · Matplotlib · Seaborn
