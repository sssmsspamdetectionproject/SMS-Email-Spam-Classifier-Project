import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation, then apply stemming
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]

    return " ".join(text)

# Load the pre-trained vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app title
st.title("SMS & Email Spam Classifier")

# Text input for the user
input_sms = st.text_area("Enter the Message (SMS)")

if st.button('Predict'):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)
    
    # Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict using the loaded model
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
