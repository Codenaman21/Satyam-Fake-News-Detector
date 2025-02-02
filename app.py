from flask import Flask, request, jsonify, render_template
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import sqlite3
import bcrypt
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import joblib

# Initialize Flask app
app = Flask(__name__ , template_folder="templates")


#enable flask session
app.secret_key = 'Dadukigadi@1234'


#function to create database in sqlite
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Call the function to create the database
init_db()

#profile handelling
@app.route('/profile', methods=['GET'])
def profile():
    if 'user_id' in session:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT name, email FROM users WHERE id = ?', (session['user_id'],))
        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({'name': user[0], 'email': user[1]}), 200
    return jsonify({'error': 'User not logged in!'}), 401

    

#register handelling
@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Store in database
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Registration successful!'}), 200
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already registered!'}), 400


#login handelling
@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, password FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user[2]):
        session['user_id'] = user[0]  # Store user session
        session['user_name'] = user[1]
        return jsonify({'message': 'Login successful!', 'name': user[1]}), 200
    else:
        return jsonify({'error': 'Invalid email or password!'}), 401


#logout handelling
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()  # Clear all session data
    return jsonify({'message': 'Logged out successfully!'}), 200



# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
word2vec_path = os.path.join(BASE_DIR, "word2vec.model")
lstm_model_path = os.path.join(BASE_DIR, "lstm_model.h5")

# Load pre-trained Word2Vec and LSTM model
word2vec_model = Word2Vec.load(word2vec_path)
lstm_model = load_model("lstm_model.keras")
print("Model loaded successfully!")

#path to save the tokenizer 
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

# Load tokenizer (ensure it is saved during training)
tokenizer = joblib.load(tokenizer_path)

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word not in stop_words])
    return text

# Route to serve the homepage
@app.route('/')
def home():
    return render_template('new2.html')  # Using your uploaded template

# API to check fake news
@app.route('/check_news', methods=['POST'])
def check_news():
    news_text = request.form['news_text']
    processed_text = preprocess_text(news_text)
    print("Processed Text:", processed_text)
    
    # Tokenize and pad sequence
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=200)
    print("Tokenized Sequence:", seq)

    # Predict using the model
    prediction = lstm_model.predict(padded_seq)
    print("Prediction Probability:", prediction)
    optimal_threshold = 0.06  # Adjust based on validation performance
    if prediction >= optimal_threshold:
        result = "Real"
    else:
        result = "Fake"


    return jsonify({'result': result, 'accuracy': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True , use_reloader=False)
