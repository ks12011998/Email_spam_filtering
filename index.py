import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import warnings
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_csv('spam.csv',encoding="latin-1")

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

X = df['v2']
y = df['v1']

cv = CountVectorizer()

X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)

clf.score(X_train,y_train)

pred= clf.predict(X_test)

import pickle as pkl
pkl.dump(clf, open("model.pkl","wb"))



