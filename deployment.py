import requests
from pprint import pprint as pp
import os
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np


def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    # Removing punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.discard("not") # removing "not" from stopwords so that it is not removed from the text
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Load Word2Vec model
model = Word2Vec.load('word2vec.model')

# Function to get vector representation of a review
def review_to_vec(review):
    vec = np.zeros(model.vector_size)
    count = 0
    for word in review:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        vec /= count
    return vec



# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account (https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml-authentication.html)
API_KEY = os.environ.get('IBM_API_KEY')
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', 
                               data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

review = input("Enter a review to be scored:\n")
review = preprocess_text(review)
review = review_to_vec(review).tolist()
input_fields = [f'vec_{i}' for i in range(100)]

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data": [{"fields": input_fields, "values": [review]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/f3e84b2a-a7a7-4e85-9f8e-acd775610ffd/predictions?version=2021-05-01', json=payload_scoring,
 headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
pp(response_scoring.json())