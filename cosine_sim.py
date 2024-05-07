from scipy.sparse import save_npz, load_npz
import numpy as np
import joblib
import string
import pandas as pd
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vector = load_npz('./DataSet/tfidf_vector_sparse.npz')
vectorizer = joblib.load('./DataSet/vectorizer.joblib')
data = pd.read_csv("./DataSet/combined_data.csv")

stop_wordsF = pd.read_csv('./DataSet/stop-words.csv')
stop_words = []
for word in stop_wordsF['0']:
    stop_words.append(word)

def process_email(user_email):
    def remove_punctuation(text):
        text = text.replace('\n', ' ') # also removing newline characters while removing punctuations
        new_text = []
        for char in text:
            if char not in string.punctuation:
                new_text.append(char)
        return ''.join(new_text)

    def convert_lower_case(text):
        new_text = []
        for char in text:
            new_text.append(char.lower())
        return ''.join(new_text)

    def remove_numbers(text):
        text_without_numbers = re.sub(r'\d', '', text)
        return text_without_numbers

    def remove_extra_space(text):
        text_without_spaces = re.sub(r'\s{1,}', ' ', text)
        text_without_spaces = text_without_spaces.strip()
        return text_without_spaces

    def tokenize(text):
        tokens = re.split('\W+', text)
        return tokens

    def lemmatization(tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in tokens]

    def stemming(tokens):
        ps = PorterStemmer()
        return [ps.stem(word) for word in tokens]

    user_email = remove_punctuation(user_email)
    user_email = convert_lower_case(user_email)
    user_email = remove_numbers(user_email)
    user_email = remove_extra_space(user_email)
    user_email = tokenize(user_email)
    user_email = lemmatization(user_email)
    user_email = stemming(user_email)
    user_email = list(filter(lambda word: word not in stop_words, user_email))

    query_text = ' '.join(user_email)
    query_vector = vectorizer.transform([query_text])

    cosine_similarities = cosine_similarity(tfidf_vector, query_vector)

    # Get indices of documents sorted by cosine similarities in descending order
    sorted_indices = np.argsort(cosine_similarities.flatten())[::-1]

    top_10_indices = sorted_indices[:10]

    # Get the cosine similarities of the top 10 documents
    top_10_cosine_similarities = cosine_similarities.flatten()[top_10_indices]

    # Retrieve the labels of the top 10 most similar documents
    spam_ham = data.loc[top_10_indices, 'label'].tolist()

    # Calculate the percentages of ham and spam labels
    ham_percentage = spam_ham.count(0) / len(spam_ham) * 100
    spam_percentage = spam_ham.count(1) / len(spam_ham) * 100

    return [ham_percentage, spam_percentage, top_10_indices.tolist(), top_10_cosine_similarities.tolist()]


def adjust_cosine_similarities(indices, cosine_similarities, relevant):
    adjustment_factor = 1.2 if relevant else 0.8
    adjusted_cosine_similarities = [similarity * adjustment_factor for similarity in cosine_similarities]

    return adjusted_cosine_similarities