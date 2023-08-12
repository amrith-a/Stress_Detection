import numpy as np
import pandas as pd
import nltk
import re
import pickle

df = pd.read_csv('stress.csv')
print(df)

df.shape
df.head()
df.describe()

nltk.download('stopwords')
stem = nltk.SnowballStemmer('english')

from nltk.corpus import stopwords
import string

sw = set(stopwords.words('english'))


def verify(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in sw]
    text = ' '.join(text)
    return text


df['text'] = df['text'].apply(verify)

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

text = ' '.join(i for i in df.text)
sws = set(STOPWORDS)
wc = WordCloud(stopwords=sws).generate(text)
plt.figure(figsize=(12, 12))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

df['label'] = df['label'].map({0: 'No stress', 1: 'Possibility of Stress'})

df = df[['text', 'label']]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

X = np.array(df['text'])
y = np.array(df['label'])
count_vect = CountVectorizer(stop_words='english')
X = count_vect.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
from sklearn.naive_bayes import BernoulliNB

demo = BernoulliNB()
demo.fit(X_train, y_train)
training_score = demo.score(X_train, y_train)
test_score = demo.score(X_test, y_test)
print(training_score)
print(test_score)

with open("Model.pickle", "wb") as file:
    pickle.dump((demo, count_vect), file)
prediction = demo.predict(X_test)


