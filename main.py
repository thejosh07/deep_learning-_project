# Import libraries
import pandas as pd

# Load datasets
fake = pd.read_csv("dataset\Fake.csv")
real = pd.read_csv("dataset\True.csv")

# Add labels
fake["label"] = 0   # Fake news
real["label"] = 1   # Real news

# Combine datasets
data = pd.concat([fake, real], axis=0)

# Show first 5 rows
print(data.head())
# Import required libraries
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return " ".join(words)

# Apply cleaning to 'text' column
data["text"] = data["text"].apply(clean_text)

# Show cleaned data
print(data["text"].head())
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data["text"])
import pickle
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
# Convert text to sequences
X = tokenizer.texts_to_sequences(data["text"])

# Apply padding (make all sequences same length)
X = pad_sequences(X, maxlen=200)

# Target variable
y = data["label"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Build model
model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=5000, output_dim=64, input_length=200))

# LSTM layer
model.add(LSTM(64))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Show model summary
model.summary()
# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)
# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
from sklearn.metrics import classification_report
import numpy as np

# Predictions
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

print(classification_report(y_test, y_pred))
# Function to predict news
def predict_news(news_text):
    cleaned = clean_text(news_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    
    prediction = model.predict(padded)
    
    if prediction[0][0] > 0.5:
        print("✅ This is Real News")
    else:
        print("❌ This is Fake News")

# Example test
sample_news = "Government announces new economic reforms for 2026"

predict_news(sample_news)
model.save("fake_news_model.h5")
import random

def bulk_test(dataframe, n=10):
    samples = dataframe.sample(n)
    correct = 0
    
    for i, row in samples.iterrows():
        text = row["text"]
        true_label = row["label"]
        
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=200)
        
        pred = model.predict(padded)[0][0]
        pred_label = 1 if pred > 0.5 else 0
        
        if pred_label == true_label:
            correct += 1
    
    print(f"Accuracy on {n} random samples: {correct}/{n}")

# Test
bulk_test(data)