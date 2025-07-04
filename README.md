# Car Rental Customer Feedback Analyzer


![IBM watsonx.ai Logo](https://www.ibm.com/brand/experience-guides/developer/8f4e3cc2b5d52354a6d43c8edbale3c9/02_8-bar_reversed.svg)

An advanced sentiment analysis system for car rental customer feedback, developed as part of the IBM GenAI with watsonx.ai certification program.

## Project Overview

This system analyzes customer feedback from car rental services using:
- **TF-IDF Vectorization** for feature extraction
- **Machine Learning models** trained from scratch
- **Sentiment classification** (Positive/Negative/Neutral)
- **Service category identification**

Key achievements:
- 65-70% classification accuracy (meeting industry standards)
- Identified Vehicle Quality as primary concern (41.7% of feedback)
- Balanced dataset with equal sentiment distribution (33.3% each)

## Implementation Details

### Data Preparation
- Collected and manually labeled 60 customer reviews from Excel/CSV
- Balanced dataset: 20 Positive, 20 Negative, 20 Neutral
- 8 service categories identified:
  - Vehicle Quality (41.7%)
  - Customer Service (15%)
  - Overall Experience (15%)
  - Process Efficiency (10%)
  - Staff Interaction (6.7%)
  - Pricing (6.7%)
  - Booking Process (3.3%)
  - Communication (1.7%)

### Libraries Used
python
# Core Libraries
import pandas as pd
import numpy as np
import re
import string
import joblib

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Web Interface
import gradio as gr
import streamlit as st

#Model Training

#Trained from scratch using:

Text Preprocessing:
Lowercasing
Punctuation removal
Whitespace normalization

#Feature Engineering:

TF-IDF Vectorization (1000 features)
Feature weights: 0.27-0.86 discriminative power

#Algorithms Implemented:

Logistic Regression (Primary)
Support Vector Machines
Naive Bayes

