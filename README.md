# SMS Spam Detection using LSTM

- SMS Spam Detection system using Deep Learning with an LSTM (Long Short-Term Memory) model. The system classifies SMS messages into two categories: "Spam" or "Not Spam". Built using TensorFlow and Keras, the model achieves 98% accuracy on the test dataset, making it highly effective for real-time SMS filtering.
---
## Features
- Text Preprocessing: Prepares data using NLTK for stopwords removal, tokenization, and other text cleaning techniques.
- Deep Learning Model: Utilizes an LSTM (Long Short-Term Memory) model to capture sequential patterns in text data, ideal for NLP tasks.
- Training: Trained on the SMS Spam Collection Dataset from Kaggle, consisting of labeled SMS messages.
- Real-Time Predictions: The system can predict whether a given SMS is spam or not using the predict_sms() function.
- Deployment Ready: The trained model and tokenizer are saved as files for easy deployment in other applications or environments.
---
## Dataset
The model is trained on the SMS Spam Collection Dataset available on Kaggle. This dataset consists of text messages that are labeled as either ham (not spam) or spam (spam). 

- **Source:** SMS Spam Collection Dataset (Kaggle)
- **Format:** CSV
- **Columns Used:**
  - `v1` → Label (`ham` or `spam`)
  - `v2` → Text message
- **Label Mapping:**
  - `ham` → `0` (Not Spam)
  - `spam` → `1` (Spam)

You can access the dataset on Kaggle [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
---
## Model Architecture
- The model is built using an LSTM (Long Short-Term Memory) network, which is ideal for sequence-based data like text. Here’s a breakdown of the architecture:
- Embedding Layer: Converts input text into dense vector representations, allowing the model to understand the semantic meaning of the words.
- SpatialDropout1D: Helps reduce overfitting by randomly dropping entire feature maps during training, ensuring the model doesn't rely too much on specific features.
- LSTM Layer: Captures sequential dependencies in the text. LSTMs are especially effective in tasks that require understanding context in text, like spam detection.
- Dense Layer: Outputs the final prediction, using a sigmoid activation function to classify the message as either spam or not spam. 
---
## Installation and Setup
- **Prerequisites**
  - To run the project, you’ll need the following software and libraries:
  -     1)Python 3.x
  -     2)TensorFlow (for deep learning model)
  -     3)Keras (for building the neural network)
  -     4)NLTK (for natural language processing)
  -     5)Pandas (for data handling)
  -     6)NumPy (for numerical operations)
---
## Installing Dependencies
- **You can install the required dependencies using the following command:**
- pip install -r requirements.txt

Make sure to clone the repository and navigate to the project directory before running the command.
---
## Running the Model
  1) Clone the repository
  2) Train the model
  3) Make Predictions

The predict_sms() function will return whether the message is spam or not.
---
## Files Generated:
 - spam_detector_lstm.h5 → Trained LSTM model
 - tokenizer.json → Tokenizer used during training
 - requirements.txt → A file containing all the Python dependencies needed to run the project.
---
## Conclusion
This project demonstrates the use of deep learning to solve a real-world problem, SMS spam detection. By leveraging an LSTM model, the system can effectively classify messages as spam or not, providing value in areas like mobile security and anti-spam filtering.
