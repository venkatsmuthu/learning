# -*- coding: utf-8 -*-
"""spam_detector.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CL_t1BEYC0EbsJtZu3By06zhlQwwRqWW
"""

!pip install nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv('mail_data.csv')
dataset.describe()
dataset.shape
print(dataset.head(5))

dataset.columns = ['Category', 'Message']
dataset['Category'] = dataset['Category'].map({'spam': 1, 'ham': 0})
print(dataset['Category'])

def pre_process(text):
  text = text.lower()
  text = re.sub(r'[^A-Za-z]',' ',text)
  words = text.split()
  words = [word for word in words if word not in stopwords.words('english')]
  stemmer = PorterStemmer()
  words = [stemmer.stem(word) for word in words]
  return ' '.join(words)

dataset['cleaned_message'] = dataset['Message'].apply(pre_process)

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(dataset['cleaned_message'])
y = dataset['Category']

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_pred,y_pred))
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))

input_test = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
#input_test_cleaned = [pre_process(text) for text in input_test]

input_test_vector = vectorizer.transform(input_test)

output_test = model.predict(input_test_vector)
print("Predicted category:", output_test)  # 1 = spam, 0 = ham