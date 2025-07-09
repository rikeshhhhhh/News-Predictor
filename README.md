# News Predictor

**News Predictor** is a machine learning web application that classifies news articles as either **Real** or **Fake** using natural language processing and ensemble learning techniques. The project is built with Python and Streamlit, and deployed on Streamlit Cloud.

---

## Live Demo

Access the deployed app here:  
[https://rikeshhhhhh-news-predictor.streamlit.app](https://rikeshhhhhh-news-predictor.streamlit.app)

---

## Project Structure

```
news-predictor/
├── app.py                 # Streamlit app script
├── news-predictor.pkl     # Trained model (Stacking Classifier)
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
```

---

## Features

- Real-time prediction of news article authenticity
- Live real news collection using RSS feeds (BBC, CNN, NYT, etc.)
- Data preprocessing with stemming and stopword removal
- TF-IDF vectorization with bigrams
- Ensemble model training: Bagging, Boosting, and Stacking
- Exploratory Data Analysis (EDA) and word cloud visualizations
- Confidence scoring for predictions

---

## Machine Learning Models

- **Bagging**: MultinomialNB with BaggingClassifier
- **Boosting**: MultinomialNB with AdaBoostClassifier
- **Stacking** (final model): MultinomialNB + RandomForest + LogisticRegression

---

## Installation

To run this project locally:

```bash
git clone https://github.com/rikeshhhhhh/News-Predictor.git
cd News-Predictor

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py
```

---

## How It Works

1. Real and fake news datasets are combined and preprocessed.
2. Text is vectorized using TF-IDF with bigrams.
3. Multiple ensemble models are trained and evaluated.
4. The Stacking Classifier with the best performance is saved.
5. The Streamlit app loads this model and predicts the authenticity of user input.

---

## Example Output

```
Input: "The Prime Minister held a meeting with the UN chief"
Prediction: Real News
Confidence: 85.67%

Input: "Covid vaccine kills millions secretly"
Prediction: Fake News
Confidence: 92.11%
```

---

## Future Improvements

- Enable CSV upload for batch prediction
- Add confidence score charts and interpretability tools
- Incorporate large language models for more nuanced predictions

---

## License

This project is licensed under the MIT License.

---

Developed by [Rikesh Pokhrel](https://github.com/rikeshhhhhh)
