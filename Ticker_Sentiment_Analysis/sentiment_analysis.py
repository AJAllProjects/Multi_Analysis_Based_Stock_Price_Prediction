import flair
import re

def clean(tweet):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    tesla = re.compile(r"(?i)@TSLA(?=\b)")
    user = re.compile(r"(?i)@[a-z0-9_]+")

    tweet = whitespace.sub(' ', tweet)
    tweet = web_address.sub('', tweet)
    tweet = tesla.sub('TSLA', tweet)
    tweet = user.sub('', tweet)

    return tweet


def sentiment_model(tweets):
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    probs = []
    sentiments = []

    tweets['text'] = tweets['text'].apply(clean)

    for tweet in tweets['text'].to_list():
        # make prediction
        sentence = flair.data.Sentence(tweet)
        sentiment_model.predict(sentence)
        # extract sentiment prediction
        probs.append(sentence.labels[0].score)
        sentiments.append(sentence.labels[0].value)

    # add probability and sentiment predictions to tweets dataframe
    tweets['probability'] = probs
    tweets['sentiment'] = sentiments

    average_sentiment = 0
    for i in range(len(probs)):
        if sentiments[i] == "NEGATIVE":
            average_sentiment += -1 * probs[i]
        else:
            average_sentiment += probs[i]

    return average_sentiment/len(probs)