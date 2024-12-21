import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model

def calculate_volatility(stock_data):
    stock_data['Return'] = 100 * stock_data['Close'].pct_change()
    stock_data.dropna(inplace=True)
    daily_volatility = stock_data['Return'].std()
    monthly_volatility = np.sqrt(21) * daily_volatility
    annual_volatility = np.sqrt(252) * daily_volatility
    return daily_volatility, monthly_volatility, annual_volatility

def fit_garch_model(returns, forecast_horizon=5):
    garch_model = arch_model(returns, p=1, q=1, mean='constant', vol='GARCH', dist='normal')
    model_fit = garch_model.fit(disp='off')
    forecast = model_fit.forecast(horizon=forecast_horizon)
    return model_fit, forecast

def rolling_predictions(returns, test_size=365):
    predictions = []
    for i in range(test_size):
        train = returns[:-(test_size - i)]
        _, forecast = fit_garch_model(train, forecast_horizon=1)
        predictions.append(forecast.variance.values[-1, :][0])
    return predictions

def volatility_predict(stock_ticker):
    stock_data = yf.Ticker(stock_ticker).history(period="10y")
    daily_volatility, monthly_volatility, annual_volatility = calculate_volatility(stock_data)
    print(f"{stock_ticker} - Daily: {daily_volatility:.2f}, Monthly: {monthly_volatility:.2f}, Annual: {annual_volatility:.2f}")
    returns = stock_data['Return']
    _, forecast = fit_garch_model(returns)
    predictions = rolling_predictions(returns)
    print(f"{stock_ticker} - 5-day Forecast: {forecast.variance.values[-1, :]}")
    return predictions

def main(tickers):
    predictions_dict = {}
    for ticker in tickers:
        predictions_dict[ticker] = volatility_predict(ticker)
    return predictions_dict

if __name__ == "__main__":
    tickers = ['TSLA', 'AAPL']
    predictions_dict = main(tickers)