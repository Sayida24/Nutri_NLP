import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_ingredients(text):
   
    # 1. Convert to string and lowercase
    text = str(text).lower()
    
    # 2. Remove punctuation and special characters using Regex
    # This keeps only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Initialize Lemmatizer (turns 'cooking' into 'cook')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # 4. Tokenize (split into words)
    words = text.split()
    
    # 5. Remove stopwords and apply Lemmatization
    # Example: "the cooked chicken" becomes "cook chicken"
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    # 6. Join words back into a single sentence
    return " ".join(cleaned_words)

if __name__ == "__main__":
    test_input = "rice, broccoli, chickens, eggs and potatoes"
# Cleaned Output will be: "cook chicken many potato"
    
    print(f"Cleaned:  {clean_ingredients(test_input)}")
    user_input = input("\nEnter food text: ")
    print(f"Original: {user_input}")
    print(f"Cleaned:  {clean_ingredients(user_input)}")