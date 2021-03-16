#Sentiment analysis 
#Author: Jáchym Putta

#Importing data handling packages
import nltk
import glob 
import pandas as pd

#Importing machine learning / mining
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB

#Loading data 
def load_review_files (filenames):
    data = []
    for filename in filenames :
        f = open(filename)
        content = f.read()
        data.append ({'filename' : filename, 'text': content })
    return data

def load_reviews (directory):
    negative_reviews = glob.glob(directory + '/neg/*.txt')
    positive_reviews = glob.glob(directory + '/pos/*.txt')

    neg = pd.DataFrame(load_review_files(negative_reviews))
    pos = pd.DataFrame(load_review_files(positive_reviews))
    neg['kind'] = 'neg'
    pos['kind'] = 'pos'

    return pd.concat([neg,pos])
  
#Import the training set
dataset_raw = load_reviews('~/nltk_data/corpora/movie_reviews')

#Format the data to tf-idf using the pipeline function, we remove stopwords along the way and introduce the Complement Naive Bayes along the way
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf', ComplementNB()),])
#Fitting the model
text_clf = text_clf.fit(dataset_raw['text'],dataset_raw['kind'])

#Prediction function using the '.predict' on our model.
def predict_sentiment(text):
    return text_clf.predict([text])
