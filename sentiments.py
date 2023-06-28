import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import ast

def load_dataset(input_file):
    return pd.read_csv(input_file)

def preprocess_and_reshape_data(df):
    # Assuming preprocessing code is already performed
    df['preprocessed_text'] = df['preprocessed_text'].apply(lambda text: ast.literal_eval(text))
    df['preprocessed_text'] = df['preprocessed_text'].apply(lambda text: ' '.join(text))
    X = df['preprocessed_text'].values.reshape(-1, 1)
    return X, df['sentiment_label']

def undersample_data(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def vectorise_text_data(X_train, X_test):
    vectoriser = TfidfVectorizer()
    X_train_vectorised = vectoriser.fit_transform(X_train.ravel())
    X_test_vectorised = vectoriser.transform(X_test.ravel())
    return X_train_vectorised, X_test_vectorised

def train_model(X_train_vectorized, y_train):
    model = LinearSVC()
    model.fit(X_train_vectorized, y_train)
    return model

def evaluate_model(model, X_test_vectorized, y_test):
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def sentiment_analysis_pipeline(input_file):
    df = load_dataset(input_file)
    X, y = preprocess_and_reshape_data(df)
    X_resampled, y_resampled = undersample_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    X_train_vectorized, X_test_vectorised = vectorise_text_data(X_train, X_test)
    model = train_model(X_train_vectorised, y_train)
    accuracy, report = evaluate_model(model, X_test_vectorised, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == '__main__':
    input_file = 'preprocessed_hotel_reviews.csv'
    sentiment_analysis_pipeline(input_file)