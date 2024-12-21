import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from ta.trend import IchimokuIndicator
import warnings
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


def get_data(ticker: str, period="10y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    ichimoku = IchimokuIndicator(high=df["High"], low=df["Low"], window1=9, window2=26, window3=52, visual=False)

    sma_3 = df["Close"].rolling(window=3).mean()
    sma_10 = df["Close"].rolling(window=10).mean()
    sma_50 = df["Close"].rolling(window=50).mean()
    sma_200 = df["Close"].rolling(window=200).mean()
    ema_10 = df["Close"].ewm(span=10, adjust=False).mean()
    ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
    volume_ema = df["Volume"].ewm(span=20, adjust=False).mean()  # Calculate EMA of Volume

    sma_3_10_cross = np.where(sma_3 > sma_10, 1, 0)
    sma_50_200_cross = np.where(sma_50 > sma_200, 1, 0)

    new_cols = {
        'ichimoku_a': ichimoku.ichimoku_a(),
        'ichimoku_b': ichimoku.ichimoku_b(),
        'sma_3': sma_3,
        'sma_10': sma_10,
        'sma_50': sma_50,
        'sma_200': sma_200,
        'ema_10': ema_10,
        'ema_50': ema_50,
        'volume_ema': volume_ema,
        'sma_3_10_cross': sma_3_10_cross,
        'sma_50_200_cross': sma_50_200_cross
    }

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.bfill().fillna(0)

    assert df.isnull().sum().sum() == 0
    return df


def get_trend(df: pd.DataFrame) -> pd.DataFrame:
    price = df["Close"].rolling(window=20).mean().bfill()
    maxima = argrelextrema(price.values, np.greater)[0]
    minima = argrelextrema(price.values, np.less)[0]
    signal = np.zeros(len(price))
    min_min = min(minima) if len(minima) > 0 else 0
    min_max = min(maxima) if len(maxima) > 0 else 0
    last_signal = 0

    if min_min < min_max:
        last_signal = 1
    elif min_min > min_max:
        last_signal = -1

    for i in range(len(price)):
        if i in maxima:
            last_signal = -1
        elif i in minima:
            last_signal = 1
        signal[i] = last_signal

    df["trend_signal"] = signal
    return df


def get_decision(df):
    signals = []
    for i in range(len(df)):
        df_row = df.iloc[i]
        trend_signal = df_row['trend_signal']

        # Simplified buy conditions for better readability and potentially better signal generation
        buy_conditions = [
            df_row['momentum_rsi'] < 30 or df_row['momentum_stoch'] < 20,
            df_row['trend_macd_diff'] > 0 and df_row['ema_10'] > df_row['ema_50'],
            df_row['volatility_atr'] < df_row['Close'] * 0.02,
            df_row['ichimoku_a'] > df_row['ichimoku_b'] and df_row['Close'] > df_row['ichimoku_a'],
            df_row['volume_obv'] > df_row['volume_ema'] and df_row['volume_cmf'] > 0.05
        ]

        # Simplified sell conditions
        sell_conditions = [
            df_row['momentum_rsi'] > 70 or df_row['momentum_stoch'] > 80,
            df_row['trend_macd_diff'] < 0 and df_row['ema_10'] < df_row['ema_50'],
            df_row['volatility_atr'] > df_row['Close'] * 0.05,
            df_row['ichimoku_a'] < df_row['ichimoku_b'] and df_row['Close'] < df_row['ichimoku_a'],
            df_row['volume_obv'] < df_row['volume_ema'] and df_row['volume_cmf'] < -0.05
        ]

        # Evaluate conditions
        if sum(buy_conditions) > sum(sell_conditions) and trend_signal == 1:
            signals.append(1)
        elif sum(sell_conditions) > sum(buy_conditions) and trend_signal == -1:
            signals.append(-1)
        else:
            signals.append(0)

        # Debugging prints
        if i % 100 == 0:  # Print every 100 rows to avoid too much output
            print(f"Index: {i}, RSI: {df_row['momentum_rsi']}, MACD Diff: {df_row['trend_macd_diff']}, EMA 10/50: {df_row['ema_10']}/{df_row['ema_50']}, Signal: {signals[-1]}")

    return signals

def plot_indicators(df, ticker):
    # Plotting closing prices and various indicators
    plt.figure(figsize=(14, 10))
    plt.subplot(411)
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['sma_10'], label='10-day SMA')
    plt.plot(df['ema_10'], label='10-day EMA')
    plt.title(f'Closing Price, SMA and EMA for {ticker}')
    plt.legend()

    plt.subplot(412)
    plt.plot(df['volume_ema'], color='orange', label='Volume EMA')
    plt.bar(df.index, df['Volume'], alpha=0.3, label='Volume')
    plt.title('Volume and Volume EMA')
    plt.legend()

    plt.subplot(413)
    plt.plot(df['ichimoku_a'], label='Ichimoku A')
    plt.plot(df['ichimoku_b'], label='Ichimoku B')
    plt.title('Ichimoku Cloud')
    plt.legend()

    plt.subplot(414)
    plt.plot(df['momentum_rsi'], color='purple', label='RSI')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(30, linestyle='--', alpha=0.5, color='green')
    plt.title('Relative Strength Index')
    plt.legend()

    plt.tight_layout()
    plt.show()

def make_prediction(signals):
    # print(signals[-30:])
    last_thirty_day_signals_average = sum(signals[-30:]) / 30
    return last_thirty_day_signals_average

def make_buy_decision(ticker="QCOM"):
    df = get_data(ticker)
    plot_indicators(df, ticker)
    df = get_trend(df)
    # print(list(df.columns))
    signals = get_decision(df)
    prediction = make_prediction(signals)
    return prediction


print(make_buy_decision())
