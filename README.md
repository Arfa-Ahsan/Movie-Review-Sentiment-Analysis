# Movie-Review-Sentiment-Analysis

### Overview
The goal of this project is to develop a machine learning model that uses natural language processing (NLP) techniques to automatically categorize movie reviews into two sentiment categories: positive and negative. This tool will enable the analysis of public opinion in real-time, helping to assess how audiences respond to movies.
This project leverages various machine learning algorithms to classify sentiment, such as Naive Bayes, Logistic Regression and Random Forest, and provides a user-friendly interface through a Streamlit web app.

### Features
- Movie Review Sentiment Classification: Automatically categorize movie reviews into positive or negative sentiment using NLP techniques.
- Machine Learning Models: Implements Logistic Regression and Random Forest models for classification.
- Real-time Analysis: The Streamlit app allows users to input movie reviews and get instant sentiment classification results.
- Customizable UI: The app includes a personalized Streamlit theme for better user experience.
  
### Installation
Follow these steps to set up and run the project locally:

##### 1. Clone the repository
```
git clone https://github.com/yourusername/Movie-Review-Sentiment-Analysis.git
cd Movie-Review-Sentiment-Analysis
```

##### 2. Create a virtual environment and activate it (optional but recommended)
```
python -m venv env
source env/bin/activate   # For Linux/macOS
env\Scripts\activate      # For Windows
```
##### 3. Install dependencies
```
pip install -r requirements.txt
```

#### 4.Run the Jupyter Notebook 
```
jupyter notebook notebooks/sentiment_analysis.ipynb
```
#### 5.Run the Streamlit app
```
streamlit run streamlit/sentiment_analysis.py
```
