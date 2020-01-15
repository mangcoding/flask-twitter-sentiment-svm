from flask import Flask, render_template, request, jsonify, flash, session, redirect, url_for
from utility import *

app = Flask(__name__)
app.secret_key = 's3cr3t'
app.debug = True

@app.route('/', methods=['POST', 'GET'])
def index():
    if isLogin():
        return redirect(url_for('home'))
    else:
        return render_template('login.html')

@app.route('/check_auth')
def check_auth():
    consumer_key = request.args.get('consumer_key')
    consumer_secret = request.args.get('consumer_secret')
    access_token = request.args.get('access_token')
    access_token_secret = request.args.get('access_token_secret')

    return jsonify(result=CheckAuth(consumer_key, consumer_secret, access_token, access_token_secret))

@app.route('/home')
def home():
    if isLogin():
        query = getQuery()
        tweets = getTweet(query["q"], query["filter"], query["r_t"], query["rpp"])
        score = getModelScore()
        return render_template('home.html', tweets=tweets, query=query, score=score, int=int, float=float, format=format, enumerate=enumerate)
    else:
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('consumer_key', None)
    session.pop('consumer_secret', None)
    session.pop('access_token', None)
    session.pop('access_token_secret', None)
    return redirect(url_for('index'))

@app.route('/set_query', methods=['POST'])
def set_query():
    sub = request.json
    setQuery(sub)
    return jsonify(result="Success")

@app.route('/add_to_dataset', methods=['POST'])
def add_to_dataset():
    sub = request.json
    if AddtoDataset(sub[0], sub[1]):
        return jsonify(result="Success")
    else:
        return jsonify(result="Fail")

@app.route('/retrain', methods=['POST'])
def retrain():
    BuildModel()
    return jsonify(result="Success")

@app.route('/tweet_detail', methods=['GET'])
def tweet_detail():
    sub = request.args.get('tweet')
    detail = TweetDetail(sub)

    return jsonify(result=detail)

@app.route('/test_predict', methods=['GET'])
def test_predict():
    s = request.args.get('sentence')
    detail = PredictSentiment(s)

    return jsonify(result=detail)


@app.route('/get_dataset')
def get_dataset():
    dataset = getDataset()
    return jsonify(dataset=dataset)

if __name__ == '__main__':
    app.run(debug=True, port=5000)