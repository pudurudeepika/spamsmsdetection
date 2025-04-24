import pandas as pd
import numpy as np
import re
import string
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D

# Download NLP data (if not downloaded)
nltk.download('all')

# Load Dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Remove unwanted columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to binary (ham → 0, spam → 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

### **Text Cleaning Function**
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize text
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply text preprocessing
data['message'] = data['message'].apply(preprocess_text)

### **Prepare Data for Deep Learning**
# Tokenization
tokenizer = Tokenizer(num_words=5000)  # Only consider top 5000 words
tokenizer.fit_on_texts(data['message'])
X = tokenizer.texts_to_sequences(data['message'])
X = pad_sequences(X, maxlen=50)  # Limit sequence length

# Split into train/test sets
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### **Define LSTM-Based Deep Learning Model**
model = Sequential([
    Embedding(5000, 128, input_length=50),  # Word embedding layer
    SpatialDropout1D(0.3),  # Helps prevent overfitting
    LSTM(100, dropout=0.3, recurrent_dropout=0.3),  # LSTM for sequence learning
    Dense(1, activation='sigmoid')  # Output layer
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### **Train Model**
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=2)

### **Evaluate Model**
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

### **Save Model for Deployment**
model.save('spam_detector_lstm.h5')
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

def predict_sms(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=50)
    prediction = model.predict(sequence)
    return "Spam" if prediction[0] > 0.5 else "Not Spam"

print(predict_sms("Win a free iPhone! Click here to claim now."))
print(predict_sms("Hey! Are we meeting at 6 PM?"))
