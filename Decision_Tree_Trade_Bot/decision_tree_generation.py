import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from ta import add_all_ta_features
import warnings
from scipy.signal import argrelextrema
from dateutil.relativedelta import relativedelta
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn import metrics


def getData(ticker: str, period = "10y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    df["SMA_3"] = df["Close"].rolling(window=3).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["SMA_3_10_cross"] = np.where(df["SMA_3"] > df["SMA_10"], 1, 0)
    df["SMA_50_200_cross"] = np.where(df["SMA_50"] > df["SMA_200"], 1, 0)
    df = df.fillna(method="bfill")
    df = df.fillna(0)
    assert df.isnull().sum().sum() == 0
    return df


def getTrend(df: pd.DataFrame) -> pd.DataFrame:
    price = df["Close"]
    price = price.rolling(window=20).mean()
    price = price.fillna(method='bfill')
    maxima = argrelextrema(price.values, np.greater)
    minima = argrelextrema(price.values, np.less)
    signal = np.zeros(len(price))
    min_min = min(minima[0])
    min_max = min(maxima[0])
    if min_min < min_max:
        lastSignal = -1
    elif min_min > min_max:
        lastSignal = 1
    else:
        raise ValueError("minima and maxima are equal")

    for i in range(len(price)):
        if i in maxima[0]:
            lastSignal = -1
        elif i in minima[0]:
            lastSignal = 1
        signal[i] = lastSignal
    df["signal"] = signal
    return df

def oneRun(model, X, df, startMoney = 10000, COMMISSION = 0.00025, lookbackarray = [1]):
    preds = model.predict(X)
    bestLookback = -1
    bestLookbackWin = -9999
    bestLookbackPortfolio = []
    for lookback in lookbackarray: # range(0, 5):
        money = startMoney
        nrStocks = 0
        portfolio = []
        for i in range(lookback, len(df)):
            if lookback > 0:
                prednow = np.median(preds[i-lookback:i+1])
            else:
                prednow = preds[i]
            if prednow == 0:
                prednow = -1
            if prednow == 1 and nrStocks == 0 and money > 10:
                howmany = money / df.iloc[i]["Close"] * .99
                cost = howmany * df.iloc[i]["Close"] * (1 + COMMISSION)
                money -= cost
                nrStocks += howmany
            elif prednow == -1 and nrStocks > 0:
                money += nrStocks * df.iloc[i]["Close"] * (1 - COMMISSION)
                nrStocks = 0
            portfolio.append(money + nrStocks * df.iloc[i]["Close"])
        money += nrStocks * df.iloc[-1]["Close"] * (1 - COMMISSION)
        win = money - startMoney
        if win > bestLookbackWin:
            bestLookback = lookback
            bestLookbackWin = win
            bestLookbackPortfolio = portfolio
    return bestLookbackWin, bestLookback, bestLookbackPortfolio

def oneRun(model, X, df, lookbackarray = [1], startMoney = 10000, COMMISSION=0.00025):
    preds = model.predict(X)
    bestLookback = -1
    bestLookbackWin = -9999
    bestLookbackPortfolio = []
    for lookback in lookbackarray:
        money = startMoney
        nrStocks = 0
        portfolio = []
        for i in range(lookback, len(df)):
            if lookback > 0:
                prednow = np.median(preds[i-lookback:i+1])
            else:
                prednow = preds[i]
            if prednow == 0:
                prednow = -1
            if prednow == 1 and nrStocks == 0 and money > 10:
                # buy
                howmany = money / df.iloc[i]["Close"] * .99
                cost = howmany * df.iloc[i]["Close"] * (1 + COMMISSION)
                money -= cost
                nrStocks += howmany
            elif prednow == -1 and nrStocks > 0:
                money += nrStocks * df.iloc[i]["Close"] * (1 - COMMISSION)
                nrStocks = 0
            portfolio.append(money + nrStocks * df.iloc[i]["Close"])
        money += nrStocks * df.iloc[-1]["Close"] * (1 - COMMISSION)
        win = money - startMoney
        if win > bestLookbackWin:
            bestLookback = lookback
            bestLookbackWin = win
            bestLookbackPortfolio = portfolio
    return bestLookbackWin, bestLookback, bestLookbackPortfolio

def getDecision(dfrow):
    if dfrow['SMA_200'] <= 57.23:
        if dfrow['trend_stc'] <= 99.89:
            if dfrow['volume_adi'] <= 513329136.0:
                if dfrow['volume_adi'] <= 72646236.0:
                    if dfrow['volume_sma_em'] <= -1.5:
                        if dfrow['trend_psar_down'] <= 48.04:
                            if dfrow['SMA_200'] <= 35.64:
                                return 0
                            else:  # if dfrow['SMA_200'] > 35.64
                                return 1
                        else:  # if dfrow['trend_psar_down'] > 48.04
                            return 1
                    else:  # if dfrow['volume_sma_em'] > -1.5
                        if dfrow['SMA_200'] <= 35.37:
                            if dfrow['volatility_atr'] <= 0.64:
                                if dfrow['volatility_atr'] <= 0.29:
                                    return 0
                                else:  # if dfrow['volatility_atr'] > 0.29
                                    if dfrow['volume_adi'] <= 72393532.0:
                                        return 1
                                    else:  # if dfrow['volume_adi'] > 72393532.0
                                        return 0
                            else:  # if dfrow['volatility_atr'] > 0.64
                                if dfrow['volatility_dch'] <= 48.07:
                                    if dfrow['volatility_bbh'] <= 42.83:
                                        if dfrow['momentum_wr'] <= -24.1:
                                            return 1
                                        else:  # if dfrow['momentum_wr'] > -24.1
                                            return 0
                                    else:  # if dfrow['volatility_bbh'] > 42.83
                                        if dfrow['volume_sma_em'] <= 5.11:
                                            return 0
                                        else:  # if dfrow['volume_sma_em'] > 5.11
                                            return 1
                                else:  # if dfrow['volatility_dch'] > 48.07
                                    return 1
                        else:  # if dfrow['SMA_200'] > 35.37
                            if dfrow['Volume'] <= 3983500.0:
                                return 0
                            else:  # if dfrow['Volume'] > 3983500.0
                                return 1
                else:  # if dfrow['volume_adi'] > 72646236.0
                    if dfrow['trend_visual_ichimoku_b'] <= 44.56:
                        if dfrow['volatility_dcw'] <= 4.94:
                            if dfrow['SMA_10'] <= 34.84:
                                return 1
                            else:  # if dfrow['SMA_10'] > 34.84
                                if dfrow['trend_adx'] <= 19.3:
                                    return 0
                                else:  # if dfrow['trend_adx'] > 19.3
                                    if dfrow['trend_macd_diff'] <= 0.08:
                                        return 1
                                    else:  # if dfrow['trend_macd_diff'] > 0.08
                                        return 0
                        else:  # if dfrow['volatility_dcw'] > 4.94
                            if dfrow['Volume'] <= 9603750.0:
                                if dfrow['momentum_pvo_signal'] <= -9.34:
                                    if dfrow['trend_ichimoku_base'] <= 19.37:
                                        if dfrow['volatility_kcc'] <= 10.46:
                                            return 0
                                        else:  # if dfrow['volatility_kcc'] > 10.46
                                            return 1
                                    else:  # if dfrow['trend_ichimoku_base'] > 19.37
                                        if dfrow['volume_nvi'] <= 1461.97:
                                            return 1
                                        else:  # if dfrow['volume_nvi'] > 1461.97
                                            if dfrow['momentum_pvo_hist'] <= -8.11:
                                                return 1
                                            else:  # if dfrow['momentum_pvo_hist'] > -8.11
                                                return 0
                                else:  # if dfrow['momentum_pvo_signal'] > -9.34
                                    if dfrow['momentum_ppo_hist'] <= -2.95:
                                        return 0
                                    else:  # if dfrow['momentum_ppo_hist'] > -2.95
                                        if dfrow['volume_sma_em'] <= 4.31:
                                            if dfrow['trend_mass_index'] <= 23.56:
                                                if dfrow['trend_adx'] <= 20.49:
                                                    return 1
                                                else:  # if dfrow['trend_adx'] > 20.49
                                                    return 0
                                            else:  # if dfrow['trend_mass_index'] > 23.56
                                                if dfrow['Close'] <= 8.15:
                                                    return 0
                                                else:  # if dfrow['Close'] > 8.15
                                                    return 1
                                        else:  # if dfrow['volume_sma_em'] > 4.31
                                            if dfrow['volatility_ui'] <= 3.88:
                                                return 0
                                            else:  # if dfrow['volatility_ui'] > 3.88
                                                return 1
                            else:  # if dfrow['Volume'] > 9603750.0
                                if dfrow['momentum_kama'] <= 9.33:
                                    return 1
                                else:  # if dfrow['momentum_kama'] > 9.33
                                    if dfrow['trend_kst'] <= -36.6:
                                        if dfrow['trend_visual_ichimoku_b'] <= 11.23:
                                            return 1
                                        else:  # if dfrow['trend_visual_ichimoku_b'] > 11.23
                                            if dfrow['trend_sma_slow'] <= 10.68:
                                                if dfrow['trend_kst_diff'] <= -16.26:
                                                    return 0
                                                else:  # if dfrow['trend_kst_diff'] > -16.26
                                                    return 1
                                            else:  # if dfrow['trend_sma_slow'] > 10.68
                                                if dfrow['volatility_bbh'] <= 29.48:
                                                    if dfrow['momentum_ppo_hist'] <= 1.22:
                                                        return 0
                                                    else:  # if dfrow['momentum_ppo_hist'] > 1.22
                                                        return 1
                                                else:  # if dfrow['volatility_bbh'] > 29.48
                                                    return 1
                                    else:  # if dfrow['trend_kst'] > -36.6
                                        if dfrow['trend_sma_fast'] <= 10.0:
                                            if dfrow['volume_nvi'] <= 4806.35:
                                                return 0
                                            else:  # if dfrow['volume_nvi'] > 4806.35
                                                return 1
                                        else:  # if dfrow['trend_sma_fast'] > 10.0
                                            if dfrow['momentum_stoch_rsi_k'] <= 0.12:
                                                if dfrow['trend_visual_ichimoku_b'] <= 10.65:
                                                    if dfrow['volume_vpt'] <= 267640.93:
                                                        return 0
                                                    else:  # if dfrow['volume_vpt'] > 267640.93
                                                        return 1
                                                else:  # if dfrow['trend_visual_ichimoku_b'] > 10.65
                                                    if dfrow['momentum_stoch_rsi_k'] <= 0.08:
                                                        return 1
                                                    else:  # if dfrow['momentum_stoch_rsi_k'] > 0.08
                                                        return 0
                                            else:  # if dfrow['momentum_stoch_rsi_k'] > 0.12
                                                if dfrow['momentum_stoch_rsi_d'] <= 0.99:
                                                    if dfrow['volume_adi'] <= 495193008.0:
                                                        return 1
                                                    else:  # if dfrow['volume_adi'] > 495193008.0
                                                        if dfrow['momentum_tsi'] <= 19.7:
                                                            return 0
                                                        else:  # if dfrow['momentum_tsi'] > 19.7
                                                            return 1
                                                else:  # if dfrow['momentum_stoch_rsi_d'] > 0.99
                                                    return 0
                    else:  # if dfrow['trend_visual_ichimoku_b'] > 44.56
                        if dfrow['volatility_bbw'] <= 5.7:
                            if dfrow['volatility_kch'] <= 47.04:
                                if dfrow['momentum_ppo'] <= -1.06:
                                    return 1
                                else:  # if dfrow['momentum_ppo'] > -1.06
                                    return 0
                            else:  # if dfrow['volatility_kch'] > 47.04
                                if dfrow['momentum_pvo_signal'] <= 7.04:
                                    if dfrow['volatility_bbw'] <= 5.44:
                                        if dfrow['trend_macd'] <= -0.34:
                                            return 0
                                        else:  # if dfrow['trend_macd'] > -0.34
                                            if dfrow['trend_vortex_ind_pos'] <= 0.88:
                                                return 1
                                            else:  # if dfrow['trend_vortex_ind_pos'] > 0.88
                                                return 1
                                    else:  # if dfrow['volatility_bbw'] > 5.44
                                        if dfrow['volatility_kcw'] <= 2.87:
                                            return 0
                                        else:  # if dfrow['volatility_kcw'] > 2.87
                                            if dfrow['trend_kst_sig'] <= -10.1:
                                                return 0
                                            else:  # if dfrow['trend_kst_sig'] > -10.1
                                                if dfrow['trend_macd_signal'] <= 0.32:
                                                    return 1
                                                else:  # if dfrow['trend_macd_signal'] > 0.32
                                                    return 0
                                else:  # if dfrow['momentum_pvo_signal'] > 7.04
                                    return 0
                        else:  # if dfrow['volatility_bbw'] > 5.7
                            if dfrow['SMA_50'] <= 42.82:
                                if dfrow['volatility_bbh'] <= 47.56:
                                    if dfrow['trend_vortex_ind_neg'] <= 1.25:
                                        if dfrow['trend_vortex_ind_pos'] <= 1.08:
                                            if dfrow['trend_mass_index'] <= 24.09:
                                                return 1
                                            else:  # if dfrow['trend_mass_index'] > 24.09
                                                return 0
                                        else:  # if dfrow['trend_vortex_ind_pos'] > 1.08
                                            return 1
                                    else:  # if dfrow['trend_vortex_ind_neg'] > 1.25
                                        return 1
                                else:  # if dfrow['volatility_bbh'] > 47.56
                                    return 1
                            else:  # if dfrow['SMA_50'] > 42.82
                                if dfrow['trend_psar_down'] <= 55.39:
                                    if dfrow['trend_adx'] <= 16.59:
                                        if dfrow['trend_visual_ichimoku_b'] <= 52.11:
                                            return 1
                                        else:  # if dfrow['trend_visual_ichimoku_b'] > 52.11
                                            return 0
                                    else:  # if dfrow['trend_adx'] > 16.59
                                        if dfrow['momentum_pvo'] <= 15.22:
                                            if dfrow['volatility_kcp'] <= 1.48:
                                                return 0
                                            else:  # if dfrow['volatility_kcp'] > 1.48
                                                if dfrow['momentum_uo'] <= 64.94:
                                                    return 1
                                                else:  # if dfrow['momentum_uo'] > 64.94
                                                    return 0
                                        else:  # if dfrow['momentum_pvo'] > 15.22
                                            if dfrow['trend_kst_sig'] <= -77.0:
                                                return 0
                                            else:  # if dfrow['trend_kst_sig'] > -77.0
                                                return 1
                                else:  # if dfrow['trend_psar_down'] > 55.39
                                    if dfrow['volatility_bbh'] <= 59.95:
                                        if dfrow['trend_visual_ichimoku_a'] <= 52.78:
                                            return 0
                                        else:  # if dfrow['trend_visual_ichimoku_a'] > 52.78
                                            return 1
                                    else:  # if dfrow['volatility_bbh'] > 59.95
                                        if dfrow['volume_nvi'] <= 1831.11:
                                            return 0
                                        else:  # if dfrow['volume_nvi'] > 1831.11
                                            if dfrow['SMA_50'] <= 60.55:
                                                if dfrow['volatility_dcw'] <= 10.16:
                                                    return 1
                                                else:  # if dfrow['volatility_dcw'] > 10.16
                                                    return 0
                                            else:  # if dfrow['SMA_50'] > 60.55
                                                if dfrow['SMA_10'] <= 60.71:
                                                    return 0
                                                else:  # if dfrow['SMA_10'] > 60.71
                                                    return 1
            else:  # if dfrow['volume_adi'] > 513329136.0
                if dfrow['trend_kst_diff'] <= 19.52:
                    if dfrow['trend_ichimoku_b'] <= 11.07:
                        if dfrow['trend_kst_diff'] <= -31.17:
                            return 0
                        else:  # if dfrow['trend_kst_diff'] > -31.17
                            return 1
                    else:  # if dfrow['trend_ichimoku_b'] > 11.07
                        return 0
                else:  # if dfrow['trend_kst_diff'] > 19.52
                    return 1
        else:  # if dfrow['trend_stc'] > 99.89
            if dfrow['SMA_200'] <= 55.15:
                if dfrow['trend_psar_up'] <= 52.59:
                    if dfrow['volume_mfi'] <= 45.32:
                        if dfrow['volume_nvi'] <= 4905.95:
                            return 0
                        else:  # if dfrow['volume_nvi'] > 4905.95
                            return 1
                    else:  # if dfrow['volume_mfi'] > 45.32
                        if dfrow['momentum_pvo'] <= -2.95:
                            return 1
                        else:  # if dfrow['momentum_pvo'] > -2.95
                            if dfrow['volume_sma_em'] <= 0.1:
                                return 0
                            else:  # if dfrow['volume_sma_em'] > 0.1
                                if dfrow['volatility_dcw'] <= 4.88:
                                    return 0
                                else:  # if dfrow['volatility_dcw'] > 4.88
                                    if dfrow['volume_nvi'] <= 1337.65:
                                        if dfrow['trend_psar_down'] <= 45.04:
                                            return 1
                                        else:  # if dfrow['trend_psar_down'] > 45.04
                                            return 0
                                    else:  # if dfrow['volume_nvi'] > 1337.65
                                        if dfrow['momentum_stoch_rsi'] <= 0.08:
                                            return 0
                                        else:  # if dfrow['momentum_stoch_rsi'] > 0.08
                                            return 1
                else:  # if dfrow['trend_psar_up'] > 52.59
                    if dfrow['volume_sma_em'] <= 4.99:
                        return 1
                    else:  # if dfrow['volume_sma_em'] > 4.99
                        return 0
            else:  # if dfrow['SMA_200'] > 55.15
                return 0
    else:  # if dfrow['SMA_200'] > 57.23
        if dfrow['momentum_stoch_rsi_k'] <= 0.05:
            if dfrow['SMA_200'] <= 62.25:
                if dfrow['SMA_200'] <= 60.56:
                    return 1
                else:  # if dfrow['SMA_200'] > 60.56
                    return 0
            else:  # if dfrow['SMA_200'] > 62.25
                return 1
        else:  # if dfrow['momentum_stoch_rsi_k'] > 0.05
            if dfrow['momentum_uo'] <= 67.46:
                if dfrow['trend_ema_slow'] <= 41.32:
                    return 0
                else:  # if dfrow['trend_ema_slow'] > 41.32
                    if dfrow['trend_mass_index'] <= 23.23:
                        if dfrow['trend_macd'] <= -1.44:
                            return 1
                        else:  # if dfrow['trend_macd'] > -1.44
                            return 0
                    else:  # if dfrow['trend_mass_index'] > 23.23
                        return 1
            else:  # if dfrow['momentum_uo'] > 67.46
                if dfrow['momentum_stoch_rsi_k'] <= 0.85:
                    return 1
                else:  # if dfrow['momentum_stoch_rsi_k'] > 0.85
                    return 0

def make_buy_decision(ticker):
    df = getData(ticker)
    decision = getDecision(df.iloc[-1])
    return decision == 1
