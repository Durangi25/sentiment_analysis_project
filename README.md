# Sentiment Analysis Project

A Flask-based sentiment analysis web app that classifies user reviews as positive or negative. The app uses a simple machine learning pipeline with text preprocessing, vocabulary vectorization, and logistic regression model inference.

## Features

- Web interface for submitting review text
- Real-time sentiment classification
- Running counts for positive and negative reviews
- Dataset-backed model training and vocabulary generation
- Modern dashboard-style UI with review history

## Project Structure

- `app.py` - Flask application and route handling
- `helper.py` - preprocessing, vectorization, model loading, and sentiment prediction
- `logger.py` - logging configuration
- `static/css/main.css` - app styling
- `templates/index.html` - web interface template
- `artifacts/sentiment_analysis.csv` - training dataset
- `static/model/` - trained model and vocabulary files

## Requirements

Install the project dependencies from `requirements.txt`. The app is built for Python 3 and includes Flask, scikit-learn, pandas, numpy, and nltk.

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Run the App

From the project root directory:

```bash
python app.py
```

Then open a browser and navigate to `http://127.0.0.1:5000`.

## How It Works

1. User submits a review through the web form.
2. `helper.py` preprocesses the text:
   - lowercase normalization
   - URL removal
   - punctuation stripping
   - digit removal
   - stopword filtering
   - stemming
3. The cleaned text is vectorized using the saved vocabulary.
4. A logistic regression model predicts sentiment.
5. The app updates the positive/negative counters and stores the review history.

## Training Data and Model

- The dataset is stored in `artifacts/sentiment_analysis.csv`.
- `helper.py` can build a vocabulary from the dataset and train a logistic regression model if the saved model file is missing or invalid.
- The model file is saved to `static/model/model.pickle`.
