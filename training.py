from sentiments import sentiment_analysis_pipeline

if __name__ == '__main__':
    input_file = 'preprocessed_hotel_reviews.csv'
    sentiment_analysis_pipeline(input_file)