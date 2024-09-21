import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import nltk
from collections import Counter
from langdetect import detect, LangDetectException

#reading data
trainingset = pd.read_csv("mediaeval-2015-trainingset.txt", sep="\t")
testingset = pd.read_csv("mediaeval-2015-testset.txt", sep="\t")

#convert to dataframe
df_train = pd.DataFrame(data=trainingset)
df_test = pd.DataFrame(data=testingset)

#PROCESSING TRAINING DATA
#------------------------
#remove retweets and reposts
rts = r"(\bRT\b|via @|repost|REPOST)"
df_train = df_train[~df_train["tweetText"].str.contains(rts)]

#remove links
links = r"(http\S+)"
df_train["tweetText"] = df_train["tweetText"].str.replace(links, "", case=False)
#^
#Ref; Code Block by User: Vishnu Kunchur, answered 2018. (online: https://stackoverflow.com/a/51994437) 

#remove mentions (e.g @username...followed by text)
mentions = r"(@\S+)"
df_train["tweetText"] = df_train["tweetText"].str.replace(mentions,"", case=False)

#remove newlines, ampersands, punctuation and other noise
signs1 = r"(&amp;|(\\n))"
signs2 = r"([.,;:'!?\"-]|[:;][)(DPO3/])"
df_train["tweetText"] = df_train["tweetText"].str.replace(signs1, "", case=False)
df_train["tweetText"] = df_train["tweetText"].str.replace(signs2, "", case=False)

#remove emojis by only keeping extended ASCII
filter_char = lambda c: ord(c) < 256
df_train["tweetText"] = df_train["tweetText"].apply(lambda s: "".join(filter(filter_char, s)))
#^
#Ref; Code block by User: xjcl, answered 2020 (online: https://stackoverflow.com/a/65109987)

#remove stopwords
stop = set(nltk.corpus.stopwords.words())
df_train["tweetText"] = df_train["tweetText"].apply(lambda x: ' '.join([w for w in x.split() if w not in (stop)]))
#^
#Ref; Code Block by User: Keiku, answered 2017. (online: https://stackoverflow.com/a/43407993)

#remove whitespace
df_train["tweetText"] = df_train["tweetText"].str.strip()

#lammatize tweetText and creating new class in the dataframe
tokenizer = nltk.tokenize.TweetTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
df_train["lemmatizedText"] = df_train["tweetText"].apply(lambda x: ' '.join([lemmatizer.lemmatize(token) for token in tokenizer.tokenize(x)]))
#^
#Ref; Code Block by User: titipata, answered 2017. (online: https://stackoverflow.com/a/47557782)

#MODEL TRAINING 
#------------------------
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#select lammatizedText and tweetText to extract features from
#select labels for training and evaluation
features_train = df_train["lemmatizedText"]
label_train = df_train["label"]
features_test = df_test["tweetText"]
label_test = df_test["label"]

#feature extraction method: Bag-of-words (BOW)
bof_vectorizer = CountVectorizer()
bof_train = bof_vectorizer.fit_transform(features_train)
bof_test = bof_vectorizer.transform(features_test)

#feature extraction method: Term Frequency - Inverse Document Frequency (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(features_train)
tfidf_test = tfidf_vectorizer.transform(features_test)

#select Multinomial Naive Bayes model
clf = MultinomialNB()

#train model on BOW features and produce classification report
clf.fit(bof_train, label_train)
predictions = clf.predict(bof_test)

print("Accuracy:", accuracy_score(label_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(label_test, predictions))
print("\nFull Report:\n", classification_report(label_test, predictions))

#train model on TF-IDF features and produce classification report
clf.fit(tfidf_train, label_train)
predictions = clf.predict(tfidf_test)

print("Accuracy:", accuracy_score(label_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(label_test, predictions))
print("\nFull Report:\n", classification_report(label_test, predictions))

#select linearSVC SVM model
clf = LinearSVC()

#train model on BOW features and produce classification report

clf.fit(bof_train, label_train)
predictions = clf.predict(bof_test)

print("Accuracy:", accuracy_score(label_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(label_test, predictions))
print("\nFull Report:\n", classification_report(label_test, predictions))

#train model on TF-IDF features and produce classification report
clf.fit(tfidf_train, label_train)
predictions = clf.predict(tfidf_test)

print("Accuracy:", accuracy_score(label_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(label_test, predictions))
print("\nFull Report:\n", classification_report(label_test, predictions))
#^
#Ref; Guide on Sentiment Analysis by User: Shaheer Khan 
#(Online: https://www.linkedin.com/pulse/sentiment-analysis-python-shaheer-khan)