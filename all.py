import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Add SVM
from sklearn.naive_bayes import MultinomialNB  # Add Naive Bayes
from sklearn.metrics import accuracy_score

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Define models
logistic_model = LogisticRegression()
svm_model = SVC()  # SVM
naive_bayes_model = MultinomialNB()  # Naive Bayes

# Define function to train selected model
def train_model(model):
    model.fit(X_train, Y_train)

# website
st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')
selected_model = st.selectbox('Select Model', ['Logistic Regression', 'SVM', 'Naive Bayes'])

# Train selected model
if selected_model == 'Logistic Regression':
    train_model(logistic_model)
elif selected_model == 'SVM':
    train_model(svm_model)
else:
    train_model(naive_bayes_model)

# Define prediction function
def prediction(input_text, model):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

# Get prediction based on selected model
if input_text:
    if selected_model == 'Logistic Regression':
        pred = prediction(input_text, logistic_model)
    elif selected_model == 'SVM':
        pred = prediction(input_text, svm_model)
    else:
        pred = prediction(input_text, naive_bayes_model)

    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')

# Display accuracy score
if selected_model == 'Logistic Regression':
    accuracy = accuracy_score(logistic_model.predict(X_test), Y_test)
elif selected_model == 'SVM':
    accuracy = accuracy_score(svm_model.predict(X_test), Y_test)
else:
    accuracy = accuracy_score(naive_bayes_model.predict(X_test), Y_test)

st.write(f'Accuracy Score: {accuracy}')
