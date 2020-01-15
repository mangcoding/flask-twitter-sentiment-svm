import tweepy
import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
import pandas as pd
import re
import json
import random
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from flask import session


pat = [
    r'http\S+',
    r'@\S+',
    r'#\S+',
    r'\\x[a-zA-Z][0-9]',
    r'\\x[a-zA-Z][a-zA-Z]',
    r'\\x[0-9][a-zA-Z]',
    r'\\x[0-9][0-9]',
    r"b\'",
    r'b\"',
    r'\\n',
] 
combined_pat = r'|'.join(pat)

norm = pd.read_csv("raw/key_norm.csv")
sw = [line.rstrip('\n') for line in open("raw/stopword")]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

filename_vectorizer = 'vectorizer.mdl'
filename_model = 'svm.mdl'

labels = ['Negative', 'Positive']

auth = None
api = None

def CleaningTweet(text):
    sent = []
    text = re.sub(combined_pat, " ", text)
    words = text.split(" ")
    for w in words:
        if w is not None:
            c = ""
            
            w = w.lower()
            text = re.sub("[^a-zA-Z]","", w)
            if norm["singkat"].isin([text]).any():
                i = norm[norm["singkat"]==text].index.values.astype(int)[0]
                text = norm["hasil"][i]
            c = text
            if c != "":
                sent.append(stemmer.stem(c))
    if len(sent) != 0:
        return " ".join(sent)


def VectorTransform(sentence):
    loaded_vectorizer = pickle.load(open("model/"+filename_vectorizer, 'rb'))
    return loaded_vectorizer.transform([sentence])

def PredictSentiment(sentence):
    loaded_model = pickle.load(open("model/"+filename_model, 'rb'))    
    sent_vec = VectorTransform(sentence)

    probability = loaded_model.predict_proba(sent_vec)
    sentiment = np.argmax(probability[0], axis=0)

    if sentiment == 1:
        s = 'Positive'
    else:
        s = 'Negative'

    return {
        'negative': probability[0][0],
        'positive': probability[0][1],
        'sentiment': s
    }

def TweetDetail(text_):
    remove_symbol = ""
    lower = ""
    remove_number = ""
    normalization = ""
    stem = ""

    sent = []
    text = re.sub(combined_pat, " ", text_)

    remove_symbol = remove_symbol + " " + text

    words = text.split(" ")
    for w in words:
        if w is not None:
            c = ""
            w = w.lower()
            
            lower = lower + " " + w

            text = re.sub("[^a-zA-Z]"," ", w)

            remove_number = remove_number + " " + text

            if norm["singkat"].isin([text]).any():
                i = norm[norm["singkat"]==text].index.values.astype(int)[0]
                text = norm["hasil"][i]

            normalization = normalization + " " + text

            if text != "":
                text =  stemmer.stem(text)
                stem = stem + " " + text
                if text not in sw:
                    sent.append(text)
    if len(sent) != 0:
        res = " ".join(sent)
    
    matrix = VectorTransform(res)

    loaded_vectorizer = pickle.load(open("model/"+filename_vectorizer, 'rb'))
    feature_names = loaded_vectorizer.get_feature_names()

    df = pd.DataFrame(matrix.T.todense(), index=feature_names, columns=['Weight'])
    df.sort_values("Weight", axis=0, ascending=False, inplace=True)
    c = df[df["Weight"] != 0]
    vec = c["Weight"].to_dict()

    return {
        'original': text_,
        'remove_symbol' : remove_symbol,
        'lower' : lower,
        'remove_number': remove_number,
        'normalization': normalization,
        'stem': stem,
        'hasil': res,
        'vec': vec
    }

def PlotConfusionMatrix(testY, hasil, name):
    cm = confusion_matrix(testY, hasil)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure("confussion_matrix")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)

    fmt = '.2f'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("static/"+ name +".png")
    plt.close("confussion_matrix")

def AddtoDataset(tweet, sentiment, dataset="dataset_clean_labelled.csv"):
    sent = CleaningTweet(tweet)
    sent_vec = VectorTransform(sent)

    with open('dataset/'+dataset,'a',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([sent ,sentiment])
        r = True

    loaded_model = pickle.load(open("model/"+filename_model, 'rb'))
    loaded_model.partial_fit(sent_vec, [sentiment])
    pickle.dump(loaded_model, open("model/"+filename_model, 'wb'))
    return r

def getDataset():
    dt_list = []
    dt = pd.read_csv("dataset/dataset_clean_labelled.csv")
    for i,x in enumerate(dt['TWEET']):
        dt_list.append({
            'tweet': x,
            'label': dt['LABEL'].iloc[i]
        })
    return dt_list

def setQuery(sub):
    query = {
        'q': sub[0],
        'filter': sub[1],
        'rpp': sub[2],
        'r_t': sub[3]
    }

    with open('model/query.json', 'w') as fp:
        json.dump(query, fp)

def getQuery():
    with open('model/query.json', 'r') as fp:
        query = json.load(fp)
    
    return query

def getModelScore():
    with open('model/score.json', 'r') as fp:
        score = json.load(fp)

    return score

def BuildModel(clean_dataset="dataset_clean_labelled.csv", ratio=0.1):
    data = pd.read_csv("dataset/"+clean_dataset)

    trainX, testX, trainY, testY = train_test_split(data["TWEET"].values.astype(str), data["LABEL"].values.astype(int), test_size=ratio)

    vectorizer = TfidfVectorizer(stop_words=sw)
    vectorizer.fit_transform(data["TWEET"].values.astype(str))

    x_train = vectorizer.transform(trainX)

    #clf = SVC(gamma='auto', probability=True)
    clf = SGDClassifier(loss='log')
    clf.fit(x_train, trainY)

    x_test = vectorizer.transform(testX)

    y_pred = np.argmax(clf.predict_proba(x_test), axis=1)

    name = "".join([chr(random.randint(97,122)) for x in range(10)])
    PlotConfusionMatrix(testY, y_pred, name)

    score = {
        'accuracy': accuracy_score(testY, y_pred),
        'precision': precision_score(testY, y_pred),
        'recall': recall_score(testY, y_pred),
        'f1': f1_score(testY, y_pred),
        'cm': name+".png"
    }

    with open('model/score.json', 'w') as fp:
        json.dump(score, fp)

    pickle.dump(clf, open("model/"+filename_model, 'wb'))
    pickle.dump(vectorizer, open("model/"+filename_vectorizer, 'wb'))

    return score

def CheckAuth(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    
    try:
        api.search()
    except tweepy.TweepError as err:
        if err.api_code == 215:
            msg = 'Bad Authentication data'
        elif err.api_code == 32:
            msg = 'Could not authenticate you'
        elif err.api_code is None:
            msg = 'No internet access'
        elif err.api_code == 25:
            msg = "Success"
            session['consumer_key'] = consumer_key
            session['consumer_secret'] = consumer_secret
            session['access_token'] = access_token
            session['access_token_secret'] = access_token_secret
        else:
            msg = err.reason[25:len(err.reason)-4]
    
    return msg

def getTweet(q, filter, r_t, rpp):
    tweets = []

    auth = tweepy.OAuthHandler(session['consumer_key'], session['consumer_secret'])
    auth.set_access_token(session['access_token'], session['access_token_secret'])
    api = tweepy.API(auth,wait_on_rate_limit=True)

    all_keyword = q.split(",")
    all_text = []
    filter = " -filter:"+filter

    for i in range(int(rpp)):
        r = random.randint(0, len(all_keyword)-1)
        q = all_keyword[r]
        q = q + filter
        for tweet in tweepy.Cursor(api.search,q=q,result_type=r_t, lang="id").items():
            if tweet.text not in all_text:
                all_text.append(tweet.text)
                tweets.append(
                    {
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'pict': tweet.user.profile_image_url_https,
                        'username': '@'+tweet.user.screen_name,
                        'name': tweet.user.name,
                        'sentiment': PredictSentiment(CleaningTweet(tweet.text))
                    }
                )
                break
    
    '''
    tweet_coba = "alhamdulillah bisa nonton konser musik yang berkualitas"
    tweets.append(
        {
            'text': tweet_coba,
            'created_at': '202020',
            'pict': "#",
            'username': '@',
            'name': "tweet.user.name",
            'sentiment': PredictSentiment(CleaningTweet(tweet_coba))
        }
    )
    '''
    
    return tweets

def isLogin():
    if 'consumer_key' in session:
        return True
    else:
        return False
