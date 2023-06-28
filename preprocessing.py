import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Removing special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Lower-casing
    text = text.lower()
    # Tokenising
    tokens = nltk.word_tokenize(text)
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatising
    lemmatiser = WordNetLemmatizer()
    tokens = [lemmatiser.lemmatize(word) for word in tokens]
    return tokens

def handle_negations(tokens):
    negation_terms = ["not", "no", "never"]
    negated = False
    result = []
    for word in tokens:
        if word in negation_terms:
            negated = not negated
        elif negated and word.isalpha():
            result.append(word + "_NEG")
        else:
            result.append(word)
    return result