import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define preprocessing functions
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Removing special characters
    cleaned_text = re.sub(r'[^\w\s]', '', ' '.join(lemmatized_tokens))

    return cleaned_text

# Apply preprocessing to each column
df = pd.read_csv('../data/raw/question_answer.csv')

df = df.drop(labels=['id'], axis=1)

for column in df.columns:
    df[column] = df[column].apply(preprocess_text)

df.to_csv('../data/processed/question_answer.csv', index=False)