import flair
import re
import yfinance as yf

stock_symbols = {
    "@TSLA": "TSLA", "$TSLA": "TSLA",
    "@AAPL": "AAPL", "$AAPL": "AAPL",
    "@GOOGL": "GOOGL", "$GOOGL": "GOOGL",
    "@MSFT": "MSFT", "$MSFT": "MSFT",
    "@AMZN": "AMZN", "$AMZN": "AMZN",
    "@FB": "FB", "$FB": "FB",
    "@NFLX": "NFLX", "$NFLX": "NFLX",
    "@INTC": "INTC", "$INTC": "INTC",
    "@AMD": "AMD", "$AMD": "AMD",
    "@NVDA": "NVDA", "$NVDA": "NVDA",
    "@ORCL": "ORCL", "$ORCL": "ORCL",
    "@CSCO": "CSCO", "$CSCO": "CSCO",
    "@IBM": "IBM", "$IBM": "IBM",
    "@SAP": "SAP", "$SAP": "SAP",
    "@TWTR": "TWTR", "$TWTR": "TWTR",
    "@UBER": "UBER", "$UBER": "UBER",
    "@LYFT": "LYFT", "$LYFT": "LYFT",
}

def verify_ticker(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    try:
        _ = ticker.info
        return True
    except:
        return False

def clean(tweet):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    potential_ticker = re.compile(r"\$[A-Za-z]+")

    tweet = whitespace.sub(' ', tweet)
    tweet = web_address.sub('', tweet)
    tweet = user.sub('', tweet)

    matches = potential_ticker.findall(tweet)
    for match in matches:
        ticker_symbol = match[1:].upper()
        if ticker_symbol not in stock_symbols and not verify_ticker(ticker_symbol):
            tweet = tweet.replace(match, '')

    return tweet

def sentiment_model(tweets):
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    probs = []
    sentiments = []

    tweets['text'] = tweets['text'].apply(clean)

    for tweet in tweets['text'].to_list():
        sentence = flair.data.Sentence(tweet)
        sentiment_model.predict(sentence)
        probs.append(sentence.labels[0].score)
        sentiments.append(sentence.labels[0].value)

    tweets['probability'] = probs
    tweets['sentiment'] = sentiments

    average_sentiment = 0
    for i in range(len(probs)):
        if sentiments[i] == "NEGATIVE":
            average_sentiment += -1 * probs[i]
        else:
            average_sentiment += probs[i]

    return average_sentiment/len(probs)
