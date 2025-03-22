# Sentiment_Analysis_NLP

# IMDB Movie Reviews Sentiment Analysis

This repository contains a task-based implementation of sentiment analysis for IMDB movie reviews using a deep learning model. The model classifies movie reviews into two categories: **positive** or **negative**, based on the text content of the review.

## Task Overview

The task involves building a machine learning pipeline for analyzing movie reviews. The following steps are part of the task:

1. **Dataset Download & Preprocessing**: Download and preprocess the IMDB dataset of 50,000 movie reviews.
2. **Model Development**: Build a deep learning model using TensorFlow/Keras to perform sentiment classification.
3. **Training**: Train the model on a training set of reviews and evaluate its performance on a test set.
4. **Testing with Custom Input**: Use the trained model to predict the sentiment of custom movie reviews.

## Requirements

- Python 3.x
- TensorFlow >= 2.x
- Pandas
- Numpy
- KaggleHub (for dataset download)

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
Dataset
The dataset used in this task is the IMDB Dataset of 50k Movie Reviews, which is available on Kaggle. You can download the dataset via KaggleHub:

Dataset: IMDB Dataset of 50k Movie Reviews

The dataset contains two columns:

review: The text of the movie review.

sentiment: The sentiment of the review, which is either positive or negative.

Task Breakdown
1. Download and Preprocess Data
python
Copy code
import kagglehub
import os
import pandas as pd

# Download dataset
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

# Load dataset
csv_path = os.path.join(path, "IMDB Dataset.csv")
df = pd.read_csv(csv_path)
2. Data Preprocessing: Encode Sentiments and Text Tokenization
python
Copy code
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Encode labels
label_map = {'positive': 1, 'negative': 0}
df['sentiment'] = df['sentiment'].map(label_map)

# Preprocess text
tokenizer = Tokenizer(num_words=10000, oov_token="")
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
3. Split Data into Training and Test Sets
python
Copy code
# Split data into training and test sets
train_size = int(len(df) * 0.8)
x_train, x_test = padded_sequences[:train_size], padded_sequences[train_size:]
y_train, y_test = df['sentiment'][:train_size], df['sentiment'][train_size:]
4. Build and Train the Model
python
Copy code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Build the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=200),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
5. Evaluate the Model
python
Copy code
# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
6. Make Predictions with Custom Reviews
python
Copy code
def predict_sentiment(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Predicted Sentiment: {sentiment} ({prediction:.4f})")

# Example usage
predict_sentiment("The film was a complete disaster, I regret watching it.")  # Expected: Negative
predict_sentiment("This movie was absolutely fantastic, I loved every moment!")  # Expected: Positive
Results
The model achieves an accuracy of 86.54% on the test set. The model is able to predict the sentiment of custom movie reviews with reasonable accuracy.

Example Outputs:
bash
Copy code
Predicted Sentiment: Negative (0.0532)
Predicted Sentiment: Positive (0.9863)
