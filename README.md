# Sentiment-Based Rating Prediction

This project utilizes Natural Language Processing (NLP) and machine learning to predict customer ratings (1 to 5) based on textual reviews. It includes a deep learning model and a Flask-based web interface for user interaction.

---

## Features

### **Data Preprocessing**
- **Text Cleaning**:
  - Handle negations (e.g., "isn't" -> "is not").
  - Remove special characters and convert text to lowercase.
- **Sentiment Analysis**:
  - Use `TextBlob` to calculate polarity scores for each review.
- **Tokenization and Padding**:
  - Convert text into numerical sequences.
  - Pad sequences to ensure uniform input length.

### **Model**
- A custom LSTM-based deep learning model:
  - **Embedding Layer**: Transforms words into dense vector representations.
  - **LSTM Layer**: Captures sequential dependencies in text.
  - **Dropout**: Prevents overfitting.
  - **Dense Layers**: Outputs the final prediction.

### **Web Interface**
- **Flask Application**:
  - Allows users to input reviews via a web form.
  - Displays:
    - Preprocessed review.
    - Sentiment polarity score.
    - Predicted rating (numerical and star representation).

---

## Installation

### **Prerequisites**
- Python 3.8 or higher.
- Required libraries:
  - TensorFlow
  - Flask
  - Pandas
  - NumPy
  - Scikit-learn
  - TextBlob

Install dependencies using:
```bash
pip install -r requirements.txt
```

### **Dataset from kaggle**
- Place the `tripadvisor_hotel_reviews.csv` file in the project directory.
- The dataset should contain:
  - **Review**: Textual reviews.
  - **Rating**: Corresponding numerical ratings.

---

## How to Use

### **1. Train the Model**
Run the script to preprocess data and train the model:
```bash
python review.py
```
- If a pre-trained model exists (`sentiment_model.h5`), it will be loaded automatically.

### **2. Run the Flask Application**
Start the web interface:
```bash
python review.py
```
- Open your browser and navigate to:
  - [http://127.0.0.1:5000](http://127.0.0.1:5000)

### **3. Test Predictions**
- Input a review.
- View the preprocessed review, sentiment score, and predicted rating.

---

## Example Reviews for Testing

### Positive:
- "The hotel was amazing! Clean rooms and friendly staff."
- "Loved my stay. Everything exceeded expectations."

### Neutral:
- "The hotel was fine, nothing special."
- "Decent service, but some issues need fixing."

### Negative:
- "Terrible experience. Dirty rooms and rude staff."
- "Never coming back. A waste of money."

---

## Technologies Used
- **Machine Learning**:
  - TensorFlow/Keras
- **NLP**:
  - TextBlob
- **Web Framework**:
  - Flask
- **Frontend**:
  - Bootstrap

---

## Future Enhancements
- Use pre-trained models like BERT for improved accuracy.
- Expand support to multiple languages.
- Deploy the application on cloud platforms (e.g., AWS, Heroku).

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributors
- salma OUERZAZI (https://github.com/OUERZAZI)
- slim ZARROUK  (https://github.com/slimzrrk)

