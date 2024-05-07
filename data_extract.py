# %%
import pandas as pd
import math
import string
import re
import operator
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import email
from tqdm import tqdm
import lxml
from bs4 import BeautifulSoup

# %%
trec = pd.read_csv('DataSet/email_origin.csv')
trec.head()

# %%
def read_email_from_file(path_file):
    try:
        with open(path_file, 'r') as file:
            message = email.message_from_file(file)
    except:
        with open(path_file, 'r', encoding='ISO-8859-1') as file:
            message = email.message_from_file(file)
    return message


def read_email_from_string(s):
    message = email.message_from_string(s)
    return message


def extract_email_body(message):
    if message.is_multipart():
        for part in message.walk():
            type_content = part.get_content_maintype()
            if type_content == 'text':
                message = part
                break
        else:
            return 'escapenonetext'

    if message.get('Content-Transfer-Encoding') == 'base64':
        try:
            body = message.get_payload(decode=True).decode()
        except:
            body = message.get_payload(decode=True).decode(encoding='ISO-8859-1')
    else:
        body = message.get_payload(decode=False)
    return body


def remove_html(s):
    soup = BeautifulSoup(s, 'lxml')
    for sp in soup(['script', 'style', 'head', 'meta', 'noscript']):
        sp.decompose()
    s = ' '.join(soup.stripped_strings)
    return s


def email_body_to_text(body):
    body = remove_html(body)
    punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
    body = re.sub('[{}]'.format(punctuation), ' ', body)
    body = re.sub('\n+', ' ', body)
    body = re.sub('\\s+', ' ', body)
    body = re.sub(r'[0-9]+', 'escapenumber', body)
    body = body.lower()
    body = re.sub(r'[a-z0-9]{20,}', 'escapelong', body)
    return body
    

# %%
trec['text'] = trec['origin'].map(read_email_from_string).map(extract_email_body).map(email_body_to_text)
trec.drop(['origin'], axis=1, inplace=True)

# %%
trec.head()

# %%
enron = pd.read_csv('DataSet/enron_spam_data.csv')
enron.head()

# %%
enron.drop(['Message ID', "Subject", "Date"], axis = 1, inplace = True)
enron.head()

# %%
enron.isna().sum()
# Removing missing values
enron.dropna(inplace = True)
enron.shape

# Checking for duplicate values
enron.duplicated().sum()
# Removing duplicate values
enron.drop_duplicates(inplace = True)
enron.shape

# Replacing ham and spam with 0 and 1 respectively
label_encoder = preprocessing.LabelEncoder() 
enron['label']= label_encoder.fit_transform(enron['Spam/Ham'])
print(enron.head())
enron.drop('Spam/Ham', axis=1, inplace=True)

# Renaming column name for merging
enron.rename(columns={'Message': 'text'}, inplace = True)
enron.head()

# %%
# Checking for missing values
trec.isna().sum()
# Checking for duplicate values
trec.duplicated().sum()

# %%
combined = pd.concat([trec, enron], ignore_index = True)
combined.info()

# %%
combined.to_csv(r"./DataSet/combined_data.csv", index=False)


