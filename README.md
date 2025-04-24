# SMS Spam Detection using LSTM

This project is a deep learning-based SMS Spam Detection system that classifies SMS messages as either **"Spam"** or **"Not Spam"**. It uses an LSTM (Long Short-Term Memory) model built with TensorFlow/Keras and achieves **98% accuracy** on test data.

---

## Features

- Preprocessing using **NLTK** (stopwords removal, tokenization, etc.)
- Built using an **LSTM model** for sequence-based text classification
- Trained on the **SMS Spam Collection Dataset**
- Real-time predictions with a simple `predict_sms()` function
- Model and tokenizer are saved for future deployment

---

## Dataset

- **Source:** SMS Spam Collection Dataset (Kaggle)
- **Format:** CSV
- **Columns Used:**
  - `v1` → Label (`ham` or `spam`)
  - `v2` → Text message
- **Label Mapping:**
  - `ham` → `0` (Not Spam)
  - `spam` → `1` (Spam)

---

## Model Architecture

1. **Embedding Layer** – Converts text into vector representation  
2. **SpatialDropout1D** – Reduces overfitting  
3. **LSTM Layer** – Captures sequential patterns in text  
4. **Dense Layer** – Sigmoid activation for binary classification  

---

## Files Generated:
spam_detector_lstm.h5 → Trained LSTM model
tokenizer.json → Tokenizer used during training

