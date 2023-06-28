# Sentiment Analysis on Hotel Reviews

This was a small NLP project with the goal of classifing hotel reviews as either positive or negative based on their sentiment.

## Prerequisites

Python 3.x along with the required packages listed in 'requirements.txt'. You can install the dependencies using pip:

```shell
pip install -r requirements.txt
```

## Usage

1. Clone the repository to your local machine:

```shell
git clone https://github.com/your-username/hotel-review-sentiment-analysis.git
```

2. Navigate to the project directory:
```shell
cd hotel-review-sentiment-analysis
```

3. Have the preprocessed hotel reviews dataset in CSV format.

4. Update the 'input_file' variable in 'training.py' to specify the path to the preprocessed hotel reviews CSV file.

5. Run the sentiment analysis pipeline by executing the following command:
'''
python training.py
'''

The pipeline will load the dataset, preprocess the data, undersample for imbalanced data, train a Linear SVC model, and evaluate its performance. The accuracy and classification report will be printed to the console.

## Project Structure

- 'preprocessing.py': contains functions for text preprocessing, incl. removing special characters, lowercasing, tokenising, removing stop words and lemmatising.
- 'sentiments.py': implements sentiment analysis functionalities, including loading the dataset, preprocessing and reshaping the data, undersampling, splitting the data, vectorising text data using TF-IDF, training a Linear SVC model and evaluating the model's performance.
- 'training.py': entry point for running the sentiment analysis pipeline. Imports the necessary functions from 'sentiments.py' and executes the pipeline.
- 'requirements.txt': lists the required packages and their versions.

## License

This project is licensed under the [MIT License](LICENSE).
