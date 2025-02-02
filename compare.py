import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
import joblib

# Ensure NLTK resources are downloaded
nltk.download('stopwords')

# Set paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
fake_csv_path = os.path.join(BASE_DIR, "Fake.csv")
true_csv_path = os.path.join(BASE_DIR, "True.csv")
word2vec_path = os.path.join(BASE_DIR, "word2vec.model")
lstm_model_path = os.path.join(BASE_DIR, "lstm_model.h5")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

# Load datasets
fake_df = pd.read_csv(fake_csv_path)
true_df = pd.read_csv(true_csv_path)

# Label data
fake_df['label'] = 0    
true_df['label'] = 1

# Downsample Fake News if it is more than Real News
if len(fake_df) > len(true_df):
    fake_df = resample(fake_df, replace=False, n_samples=len(true_df), random_state=42)
elif len(true_df) > len(fake_df):
    true_df = resample(true_df, replace=False, n_samples=len(fake_df), random_state=42)


# Combine and preprocess
data = pd.concat([fake_df, true_df], ignore_index=True)
texts = data['text']
labels = data['label']

# Preprocessing function
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word not in stop_words])
    return text

# Apply preprocessing
texts = texts.apply(preprocess_text)

# Train Word2Vec
tokenized_texts = [text.split() for text in texts]
word2vec_model = api.load("word2vec-google-news-300")

# Tokenizer for LSTM
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
joblib.dump(tokenizer, tokenizer_path)  # Save tokenizer
print("Tokenizer saved successfully!")

X_sequences = tokenizer.texts_to_sequences(texts)
X_padded = pad_sequences(X_sequences, maxlen=200)
X_train, X_test, y_train, y_test = train_test_split(X_padded, labels, test_size=0.25, random_state=42)

# Build LSTM Model
lstm_model = Sequential([
    Embedding(input_dim=10000, output_dim=300),
    LSTM(64, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Predict on Test Data
lstm_model.fit(X_train, y_train, epochs=8, batch_size=64, validation_split=0.2)

# Now predict on Test Data
y_pred = (lstm_model.predict(X_test) > 0.5).astype(int)


# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()


# Save Model
lstm_model.save("lstm_model.keras")
print("Model saved successfully!")
