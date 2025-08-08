# Customer Feedback Analyzer for Car Rental Services

An advanced sentiment analysis system designed to process and analyze customer feedback for car rental services. Developed as part of the **IBM GenAI with watsonx.ai certification program**, this project leverages machine learning to classify sentiments and identify key service categories from customer reviews.

## Table of Contents

- Project Overview
- Features
- Installation
- Usage
- Implementation Details
- Dataset
- Technologies Used
- Results
- Contributing
- License
- Contact

## Project Overview

The **Customer Feedback Analyzer** is a machine learning-based system that processes car rental customer feedback to:

- Classify sentiments as **Positive**, **Negative**, or **Neutral**.
- Identify key service categories mentioned in reviews (e.g., Vehicle Quality, Customer Service).
- Provide actionable insights for car rental businesses to improve service quality.

The system achieves **65-70% classification accuracy**, aligning with industry standards, and identifies **Vehicle Quality** as the primary concern in 41.7% of feedback.

## Features

- **Sentiment Analysis**: Classifies customer feedback into Positive, Negative, or Neutral sentiments.
- **Service Category Identification**: Detects specific service areas (e.g., Vehicle Quality, Pricing) mentioned in reviews.
- **Interactive Web Interface**: Built with Gradio and Streamlit for easy feedback analysis.
- **Custom-Trained Models**: Uses Logistic Regression and Naive Bayes for robust sentiment classification.
- **Balanced Dataset**: Ensures equal distribution of sentiments (33.3% each) for unbiased model training.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Agrannya-Singh/Customer-Feedback-Analyzer.git
   cd Customer-Feedback-Analyzer
   ```

2. **Install Dependencies**: Ensure you have Python 3.8+ installed. Then, install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**: The dataset is included in the repository as an Excel/CSV file. Ensure it is placed in the project root or update the file path in the code.

4. **Optional: Web Interface Setup**: To use the Gradio or Streamlit interface, ensure the respective libraries are installed:

   ```bash
   pip install gradio streamlit
   ```

## Usage

1. **Run the Sentiment Analysis**: Execute the main script to preprocess data, train models, and analyze feedback:

   ```bash
   python main.py
   ```

2. **Launch the Web Interface**:

   - For Gradio:

     ```bash
     python app_gradio.py
     ```
   - For Streamlit:

     ```bash
     streamlit run app_streamlit.py
     ```

3. **Input Feedback**: Use the web interface or script to input customer reviews and view sentiment and category predictions.

## Implementation Details

### Data Preprocessing

- **Text Cleaning**: Lowercasing, punctuation removal, and whitespace normalization.
- **Feature Extraction**: TF-IDF Vectorization with 1000 features, achieving feature weights between 0.27 and 0.86 for discriminative power.

### Machine Learning Models

- **Logistic Regression**: Primary model for sentiment classification.
- **Naive Bayes**: Secondary model for comparison and validation.
- **Training**: Models trained on a balanced dataset with train-test split using `sklearn`.

### Web Interface

- **Gradio**: Provides an interactive UI for real-time feedback analysis.
- **Streamlit**: Offers a dashboard-style interface for visualizing results.

## Dataset

The dataset consists of **60 manually labeled customer reviews** stored in Excel/CSV format, with:

- **Balanced Sentiment Distribution**: 20 Positive, 20 Negative, 20 Neutral reviews (33.3% each).
- **Service Categories**:
  - Vehicle Quality (41.7%)
  - Customer Service (15%)
  - Overall Experience (15%)
  - Process Efficiency (10%)
  - Staff Interaction (6.7%)
  - Pricing (6.7%)
  - Booking Process (3.3%)
  - Communication (1.7%)

## Technologies Used

- **Python Libraries**:
  - `pandas`, `numpy`: Data manipulation
  - `re`, `string`: Text preprocessing
  - `joblib`: Model serialization
  - `scikit-learn`: Machine learning (TfidfVectorizer, LogisticRegression, SVC, MultinomialNB, train_test_split)
  - `gradio`, `streamlit`: Web interfaces
- **Development Environment**: Python 3.8+
- **Version Control**: Git

## Results

- **Classification Accuracy**: 65-70%, meeting industry standards.
- **Primary Concern**: Vehicle Quality, mentioned in 41.7% of feedback.
- **Model Performance**: Logistic Regression outperforms Naive Bayes in most cases due to better handling of TF-IDF features.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please ensure your code follows the project's coding standards and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or feedback, reach out to the project maintainer:

- **GitHub**: Agrannya-Singh
- **Email**: \[singh.agrannya@gmail.com\]

---

© 2025 Agrannya Singh. Built as part of the IBM GenAI with watsonx.ai certification program.

  <img src="https://raw.githubusercontent.com/Agrannya-Singh/Customer-Feedback-Analyzer/05f43e6a44b15cc617f5cdabda593635a295eecc/mermaid-flow.svg" height='50%'>

