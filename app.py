#!pip install gradio 
import gradio as gr
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string
import numpy as np



# Load data
df = pd.read_csv('car_rental_feedback_sentiment.csv - Copy (1).csv')  # Make sure filename matches

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('\s+', ' ', text).strip()
    return text

# Check if required columns exist
if not all(col in df.columns for col in ['review', 'sentiment_value']):
    raise ValueError("CSV file must contain 'review' and 'sentiment_value' columns")

df['cleaned_review'] = df['review'].apply(preprocess_text)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment_value']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

def predict_sentiment(text):
    try:
        cleaned_text = preprocess_text(text)
        text_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        confidence = probabilities.max()
        sentiment_label = 'Positive' if prediction == 1 else 'Negative' if prediction == -1 else 'Neutral'
        return f"{sentiment_label} (confidence: {confidence*100:.1f}%)"
    except Exception as e:
        return f"Error processing your input: {str(e)}"

description = """
<div style="display: flex; align-items: center; gap: 24px; background: #fff; padding: 16px 32px; border-radius: 14px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: fit-content;">
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/c/c5/Vellore_Institute_of_Technology_seal_2017.svg/300px-Vellore_Institute_of_Technology_seal_2017.svg.png" alt="VIT Logo" style="height: 48px; width: auto; border-radius: 8px; background: #f5f7fa; padding: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.05); transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.08)'" onmouseout="this.style.transform='scale(1)'"/>
  <img src="https://www.ibm.com/brand/experience-guides/developer/8f4e3cc2b5d52354a6d43c8edba1e3c9/02_8-bar-reverse.svg" alt="IBM Logo" style="height: 48px; width: auto; border-radius: 8px; background: #f5f7fa; padding: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.05); transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.08)'" onmouseout="this.style.transform='scale(1)'"/>
</div>
<br>
<b style="font-size: 1.3rem; letter-spacing: 1px; color: #222; font-family: 'Segoe UI', Arial, sans-serif;">
  Car Rental Feedback Analyzer
</b>
<p>Enter your car rental review below. The app will predict the sentiment (Positive, Neutral, Negative).</p>
"""

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter car rental review", placeholder="Type your car rental experience here..."),
    outputs=gr.Textbox(label="Sentiment Prediction"),
    title="🚗 Car Rental Feedback Analyzer",
    description=description,
    examples=[
        ["The car was clean and the staff was friendly"],
        ["Terrible experience with late delivery"],
        ["Average service, nothing special"]
    ]
)

# Test locally in Colab first
iface.launch()