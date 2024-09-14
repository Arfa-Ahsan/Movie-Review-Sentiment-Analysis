#importing necessary libraries
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  


st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge_1yB5, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK, .github-icon-class {  /* Add the class for GitHub icon */
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load the saved models 
model_paths = {
    'Logistic Regression': 'Models/logistic_regression_model.pkl',
    'Naive Bayes':  'Models/naive_bayes_model.pkl',
    'Random Forest': 'Models/random_forest_model.pkl',
}

vectorizer_path = 'Models/tfidf_vectorizer.pkl'

# Load the vectorizer
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the predict function
def predict(text, model, vectorizer):
    # Transform the input text to vector form
    text_transformed = vectorizer.transform([text])
    
    # Make the prediction and get the probability
    prediction = model.predict(text_transformed)[0]
    probability = model.predict_proba(text_transformed)[0]

    # Determine sentiment and corresponding emoji
    sentiment = 'Positive Sentiment 😊' if prediction == 1 else 'Negative Sentiment 😔'
    
    return sentiment, probability


st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Sentiment Analysis App</h1>
    </div>
    """, unsafe_allow_html=True
)
st.image('Images/sentiment pic.png')

st.info('Enter the information below:')

# Text input
text = st.text_area('Enter Movie Review:')
options = list(model_paths.keys())  
selected = st.multiselect('Select a Suitable Model', options)
st.caption('The logistic regression have the highest accuracy.')

# Remove stopwords from the text for visualizations
stop_words = set(stopwords.words('english'))
stop_words.update(['movie', 'film', 'cinema', 'films', 'character','characters']) 
words = [word for word in text.split() if word.lower() not in stop_words]
cleaned_text = ' '.join(words)

# Sidebar for visualizations
with st.sidebar:
    st.markdown(
    """
    <div style='text-align: center;'>
        <h2>Visualizations</h2>
    </div>
    """, unsafe_allow_html=True
)

    st.info('Enter the review to make the visualizations appear')
    if text:
        # Word cloud option
        if st.checkbox("Show Word Cloud"):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        
        # Word frequency
        word_counts = Counter(words)
        common_words = word_counts.most_common(5)
        
        # Bar chart option
        if st.checkbox("Show Bar Chart"):
            st.write("**Top 5 Most Common Words**")
            word_labels, word_values = zip(*common_words)
            plt.figure(figsize=(10, 5))
            plt.bar(word_labels, word_values, color='skyblue')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        
        # Pie chart option
        if st.checkbox("Show Pie Chart"):
            st.write("**Word Distribution**")
            word_labels, word_values = zip(*common_words)
            plt.figure(figsize=(7, 7))
            plt.pie(word_values, labels=word_labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            st.pyplot(plt)

# Predict button action
if st.button('Predict Sentiment') and text:
    if selected:
        # Load models only when needed
        for model_name in selected:
            with open(model_paths[model_name], 'rb') as model_file:
                model = pickle.load(model_file)
            
            # Get prediction and probabilities
            sentiment, probability = predict(text, model, vectorizer)
            
            # Display results using markdown for better layout
            st.markdown(f"<div style='border:1px solid #ddd; padding: 10px; margin-bottom: 15px;'>"
                        f"<h3>{model_name}</h3>"
                        f"<p><strong>Prediction:</strong> {sentiment}</p>"
                        f"<p><strong>Probabilities:</strong> Positive - {probability[1]:.2f}, Negative - {probability[0]:.2f}</p>"
                        "</div>", unsafe_allow_html=True)

            # Visualization of sentiment probabilities
            sentiment_labels =  ['Negative', 'Positive']
            sentiment_values = [probability[0], probability[1]]
            
            # Bar chart for sentiment distribution
            plt.figure(figsize=(3, 2))
            plt.bar(sentiment_labels, sentiment_values, color=['#d031e8','#f5c7fc'])
            plt.ylabel('Probability')
            plt.title(f'Sentiment Probability Distribution',fontsize=10)
            st.pyplot(plt)

    else:
        st.write("Please select at least one model.")
