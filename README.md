# SMS Spam Detection using LSTM

This is a deep learning-based SMS Spam Detection project using an LSTM (Long Short-Term Memory) model built with TensorFlow/Keras. The model classifies SMS messages as either "Spam" or "Not Spam" with a high accuracy (98%).

------------------------------------------------------------
Features:
------------------------------------------------------------
- Preprocessing with NLTK (stopword removal, tokenization, etc.)
- LSTM model for sequential text classification
- Real-time predictions using a custom function
- Exported model and tokenizer for future deployment
- Achieves ~98% accuracy on test data

------------------------------------------------------------
Dataset:
------------------------------------------------------------
- Source: SMS Spam Collection Dataset (UCI / Kaggle)
- Format: CSV
- Label Mapping: 
    - "ham" → 0 (Not Spam)
    - "spam" → 1 (Spam)

------------------------------------------------------------
Libraries Required:
------------------------------------------------------------
Install dependencies via:
> pip install -r requirements.txt

Contents of requirements.txt:
pandas
numpy
nltk
tensorflow
scikit-learn

------------------------------------------------------------
Model Architecture:
------------------------------------------------------------
1. Embedding Layer – Converts words to vector representations
2. SpatialDropout1D – Prevents overfitting
3. LSTM Layer – Learns sequence patterns
4. Dense Output Layer – Sigmoid activation for binary classification

------------------------------------------------------------
How to Use:
------------------------------------------------------------
1. Clone the repository:
   git clone https://github.com/your-username/spam-sms-lstm.git
   cd spam-sms-lstm

2. Run the model training and test:
   python spam_detector_lstm.py

3. Example usage of prediction function:
   print(predict_sms("Win a free iPhone! Click here to claim now."))
   # Output: Spam

   print(predict_sms("Hey! Are we meeting at 6 PM?"))
   # Output: Not Spam

------------------------------------------------------------
Files Generated:
------------------------------------------------------------
- spam_detector_lstm.h5       → Trained LSTM model
- tokenizer.json              → Tokenizer used during training

------------------------------------------------------------
License:
------------------------------------------------------------
This project is licensed under the MIT License. See LICENSE file for more details.

------------------------------------------------------------
Credits:
------------------------------------------------------------
- SMS Spam Dataset: UCI / Kaggle
- Libraries: TensorFlow, Keras, NLTK, Scikit-learn
