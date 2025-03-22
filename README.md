# IMDb Sentiment Analysis

A sentiment analysis script using Natural Language Processing (NLP) to classify movie reviews as positive or negative based on the IMDb dataset.

## Description

This script performs sentiment analysis on movie reviews using a neural network model built with TensorFlow and Keras. The model is trained on the IMDb dataset consisting of 50,000 movie reviews labeled as either "positive" or "negative."

### Key Features:
- **Data Preprocessing**: Text data is cleaned and tokenized.
- **Model Architecture**: Uses an LSTM-based deep learning model to classify sentiment.
- **Custom Prediction**: Includes a `predict_sentiment()` function for making predictions on custom input.

## Requirements

- Python 3.x
- TensorFlow
- Pandas
- Numpy
- KaggleHub (for downloading the dataset)
