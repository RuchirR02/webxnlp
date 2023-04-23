from flask import Flask,render_template,request, session, redirect, url_for
from flask_pymongo import PyMongo
import pickle
import numpy as np
import bcrypt
import pandas as pd
import pymongo

import pandas as  pd
import spacy
    
import seaborn as sns
import string

from tqdm import tqdm
from textblob import TextBlob
    
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
   
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
    
    
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
    
import swifter




df = pd.read_json("News_Category_Dataset_v3.json", lines= True)
df.to_csv("nlp.csv", header=True, index = False)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/Explore')
def recommend_nlp():
    return render_template('Explore.html')


@app.route('/Explore_news',methods=['post'])
def recommend_cont():
    user_input = request.form.get('user_input')
    
    stop_words_ = set(stopwords.words('english'))
    wn = WordNetLemmatizer()
    my_sw = ['make', 'amp',  'news','new' ,'time', 'u','s', 'photos',  'get', 'say']

    def black_txt(token):
        return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2 and token not in my_sw
    
    def clean_txt(text):
        clean_text = []
        clean_text2 = []
        text = re.sub("'", "",text)
        text=re.sub("(\\d|\\W)+"," ",text)    
        clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
        clean_text2 = [word for word in clean_text if black_txt(word)]
        return " ".join(clean_text2)
    def subj_txt(text):
        return  TextBlob(text).sentiment[1]

    def polarity_txt(text):
        return TextBlob(text).sentiment[0]

    def len_text(text):
        if len(text.split())>0:
            return len(set(clean_txt(text).split()))/ len(text.split())
        else:
            return 0
    df['text'] = df['headline']  +  " " + df['short_description']

    df['text'] = df['text'].swifter.apply(clean_txt)
    df['polarity'] = df['text'].swifter.apply(polarity_txt)
    df['subjectivity'] = df['text'].swifter.apply(subj_txt)
    df['len'] = df['text'].swifter.apply(lambda x: len(x))

    X = df[['text', 'polarity', 'subjectivity','len']]
    y =df['category']

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y)
    v = dict(zip(list(y), df['category'].to_list()))

    text_clf = Pipeline([
     ('vect', CountVectorizer(analyzer="word", stop_words="english")),
     ('tfidf', TfidfTransformer(use_idf=True)),
     ('clf', MultinomialNB(alpha=.01)),
     ])
    
    text_clf.fit(x_train['text'].to_list(), list(y_train))

    import pickle
    with open('model.pkl','wb') as f:
        pickle.dump(text_clf,f)

    with open('model.pkl', 'rb') as f:
        clf2 = pickle.load(f)
    docs_new = [user_input]
    predicted = clf2.predict(docs_new)
    cat = v[predicted[0]]
    cat_news =  pd.read_csv('nlp.csv', index_col = "category")

    og_news = cat_news.loc[cat]
    head = og_news['headline']
    des =  og_news['short_description']
    date = og_news['date']
    read_more = og_news['link']
    data = []

    for i in range(28):
        item = []
        item.append(head[i])
        item.append(des[i])
        item.append(date[i])
        item.append(read_more[i])
        data.append(item)
 

    
    return render_template('Explore.html',data = data)


if __name__ == '__main__':
    app.run(debug=True)