# Satyam-Fake-News-Detector

Satyam is an advanced web-based application designed to detect fake news using Natural Language Processing (NLP) techniques. In today's digital world, misinformation spreads rapidly, leading to severe consequences for society. Satyam aims to combat this issue by providing a reliable and efficient solution for verifying the authenticity of news articles.

This project is built with a powerful combination of technologies: a Python backend for processing data, an HTML/CSS/JavaScript frontend for an interactive user experience, and an SQLite database to store user credentials and news verification history. By leveraging machine learning models and NLP techniques, Satyam accurately classifies news articles as real or fake, helping users make informed decisions about the information they consume.

## Overview

Satyam is a web-based application designed to detect fake news using NLP techniques. The project features a Python-based backend, an HTML/CSS/JavaScript frontend, and an SQLite database for user authentication. It provides an intuitive interface where users can submit news articles, analyze their authenticity, and store results for future reference.

## Project Team: Scriptifiers
**Satyam** was developed by **Scriptifiers**, a team of four dedicated developers:
- **Naman Sachdeva**
- **Tarun Barkoti**
- **Yash Mohite**
- **Aniket Parhashar**

Together, we collaborated to build an accurate and user-friendly fake news detection system using cutting-edge NLP techniques.

## Features

- **User Authentication**: Users can securely register, log in, and manage their accounts.
- **Fake News Detection**: Uses advanced NLP models to classify news articles as real or fake with high accuracy.
- **Database Storage**: SQLite stores user credentials, login details, and past analyses.
- **Responsive UI**: Developed using HTML, CSS, and JavaScript for an intuitive and visually appealing experience.
- **News Submission**: Users can input text-based news articles or provide URLs for verification.
- **Prediction System**: The system displays classification results along with confidence scores to help users understand the model's accuracy.
- **History Tracking**: Users can track and review their past news verification attempts and results.
- **Admin Dashboard**: Administrators can monitor user activity, track trends in fake news submissions, and manage database entries.
- **API Support**: Offers REST API endpoints for seamless integration with external applications and platforms.
- **Multi-Language Support**: Planned future enhancement to support multiple languages for broader accessibility.
- **Interactive Reports**: Generates comprehensive reports on fake news trends based on user-submitted data.
- **User Feedback System**: Enables users to report false positives or negatives, improving model performance over time.

## Tech Stack

- **Backend**: Python (Flask/Django) for handling requests, processing data, and running the NLP model.
- **Frontend**: HTML, CSS, JavaScript for designing an interactive and responsive user interface.
- **Database**: SQLite for secure storage of user credentials, news data, and history logs.
- **NLP Model**: Uses libraries such as Scikit-learn, TensorFlow, and Hugging Face Transformers for text classification.
- **REST API**: Enables external applications to utilize Satyam’s functionalities through API requests.


## Workflow of the Project
### Step 1: User Registration & Login
- Users register by providing a username, email, and password.
- Login credentials are securely stored in the SQLite database.
- Users authenticate using their credentials to access the system.

### Step 2: News Submission
- Users can enter a news article as plain text or provide a URL.
- The system extracts relevant content for analysis.

### Step 3: Preprocessing & NLP Analysis
- Text is cleaned by removing stopwords, special characters, and redundant data.
- The NLP model (BERT, RoBERTa, or another ML model) tokenizes and processes the input.
- The model predicts whether the news is **real** or **fake**.

### Step 4: Displaying Results
- The system returns the classification result with a confidence score.
- Users receive visual feedback indicating the likelihood of the news being fake or real.

### Step 5: History Tracking
- The result is saved in the database for future reference.
- Users can view their previous analyses and track patterns over time.

### Step 6: Admin Monitoring & API Integration
- Admins can oversee user activity and manage database records.
- External applications can integrate with Satyam through provided API endpoints.

### Step 7: User Feedback & Model Improvement
- Users can provide feedback on incorrect classifications.
- Feedback is used to retrain and enhance the NLP model.


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Codenaman21/Satyam-Fake-News-Detector.git
   cd satyam
   ```
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run database migrations:**
   ```bash
   python setup_db.py  # Ensures SQLite is set up with required tables
   ```
5. **Start the application:**
   ```bash
   python app.py
   ```
6. **Access the web app:**
   Open `http://127.0.0.1:5000/` in your browser.

## Usage

1. **Register/Login**: Users must sign up or log in to access the fake news detection system.
2. **Submit News**: Users can enter a news article or provide a URL for analysis.
3. **View Results**: The NLP model classifies the news as fake or real and provides a confidence score.
4. **Review History**: Users can track past analyses and verify previous results.
5. **API Access**: Developers can integrate Satyam’s functionalities using provided API endpoints.
6. **Admin Dashboard**: Administrators can monitor usage statistics, manage users, and enhance detection capabilities.

## Database Structure

- `users` table:
  - `id` (Primary Key)
  - `username`
  - `password_hash`
  - `email`
- `news` table:
  - `id` (Primary Key)
  - `user_id` (Foreign Key to `users`)
  - `news_content`
  - `result` (Fake/Real)
- `history` table:
  - `id` (Primary Key)
  - `user_id` (Foreign Key to `users`)
  - `timestamp`
  - `news_id` (Foreign Key to `news`)

## Future Enhancements

- **Improve NLP Model**: Integrate deep learning techniques such as transformers to increase accuracy.
- **User Feedback System**: Allow users to report incorrect classifications and provide corrections.
- **REST API Expansion**: Develop more robust API endpoints for greater flexibility and external integrations.
- **Multi-language Support**: Expand capabilities to detect fake news in multiple languages for a global audience.
- **Advanced Data Visualization**: Incorporate analytics dashboards displaying fake news trends over time.
- **Mobile App Development**: Create a dedicated mobile application for better accessibility.
- **Social Media Integration**: Enable direct news verification from social media platforms.


## Contributing

Contributions are welcome! If you would like to improve Satyam, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the **APACHE 2.O** License.

## Contact

For any issues, feature requests, or suggestions, please email `nsachdeva300@gmail.com` or raise an issue in the GitHub repository.

