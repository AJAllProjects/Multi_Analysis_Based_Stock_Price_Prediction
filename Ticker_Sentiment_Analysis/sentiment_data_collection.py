import Ticker_Sentiment_Analysis.constants as constants
from datetime import datetime, timedelta
import requests
import pandas as pd

def get_data(tweet):
    data = {
        'id': [tweet['id']],
        'created_at': [tweet['created_at']],
        'text': [tweet['text']]
    }

    return data

def define_params(query = "", tweet_mode = "extended", language = "en", count = "500"):
    if type(count) != str:
        count = str(int(count))

    params = {
        'q': query,
        'tweet_mode': tweet_mode,
        'lang': language,
        'count': count
    }

    return params

def time_travel(now, mins, dtformat = '%Y-%m-%dT%H:%M:%SZ'):
    now = datetime.strptime(now, dtformat)
    back_in_time = now - timedelta(minutes=mins)
    return back_in_time.strftime(dtformat)

def retrieve_requested_data(query = '(tesla OR tsla OR elon musk) (lang:en)'):
    endpoint = 'https://api.twitter.com/2/tweets/search/recent'
    headers = {'authorization': f'Bearer {constants.TWITTER_BEARER_TOKEN}'}
    params = {
        'query': query,
        'max_results': '100',
        'tweet.fields': 'created_at,lang'
    }

    dtformat = '%Y-%m-%dT%H:%M:%SZ'

    now = datetime.now()
    last_week = now - timedelta(days=7)
    now = now.strftime(dtformat)

    df = pd.DataFrame(columns=['id',
        'created_at',
        'text'])

    while True:
        if datetime.strptime(now, dtformat) < last_week:
            break

        pre60 = time_travel(now, 60)
        params['start_time'] = pre60
        params['end_time'] = now
        response = requests.get(endpoint,
                                params=params,
                                headers=headers)
        now = pre60
        try:
            response.json()['data']
            for tweet in response.json()['data']:
                row = get_data(tweet)
                df = pd.concat([df, pd.DataFrame(row)], axis=0, join='outer', ignore_index=True)
        except:
            continue

    return df


