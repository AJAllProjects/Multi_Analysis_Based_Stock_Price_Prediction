import yfinance as yf
import math
import numpy as np
import pandas as pd
from arch import arch_model
from arch.__future__ import reindexing

def volatility_predict(stock_ticker = "TSLA"):
    ticker = stock_ticker
    tsla_data = yf.Ticker(ticker).history(period="10y")


    tsla_data['Return'] = 100 * (tsla_data['Close'].pct_change())

    tsla_data.dropna(inplace=True)

    daily_volatility = tsla_data['Return'].std()

    monthly_volatility = math.sqrt(21) * daily_volatility

    annual_volatility = math.sqrt(252) * daily_volatility


    garch_model = arch_model(tsla_data['Return'], p = 1, q = 1,
                          mean = 'constant', vol = 'GARCH', dist = 'normal')

    gm_result = garch_model.fit(disp='off')

    gm_forecast = gm_result.forecast(horizon = 5)

    rolling_predictions = []
    test_size = 365

    for i in range(test_size):
        train = tsla_data['Return'][:-(test_size - i)]
        model = arch_model(train, p=1, q=1)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

    rolling_predictions = pd.Series(rolling_predictions, index=tsla_data['Return'].index[-365:])

    return rolling_predictions, monthly_volatility, annual_volatility, daily_volatility, gm_forecast

if __name__ == "__main__":
    print(volatility_predict()[3])