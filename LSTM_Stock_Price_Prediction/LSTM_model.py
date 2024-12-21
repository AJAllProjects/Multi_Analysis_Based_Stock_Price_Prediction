import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from keras_tuner import BayesianOptimization, Objective
import os
import shutil
import gc
import keras.backend as K
import logging
import tensorflow as tf
from joblib import Parallel, delayed

# Enable GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to calculate EMA using vectorized operations
def calculate_ema(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    ema = np.convolve(values, weights, mode='valid')
    return np.concatenate([np.full(window-1, ema[0]), ema])

# Function to add technical indicators with vectorized operations
def add_technical_indicators(data, ema_window, rsi_window, macd_short, macd_long, macd_signal):
    data['EMA'] = calculate_ema(data['Close'].values, ema_window)
    delta = data['Close'].diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_window).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_window).mean().values
    rs = avg_gain / avg_loss
    data['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    exp1 = data['Close'].ewm(span=macd_short, adjust=False).mean()
    exp2 = data['Close'].ewm(span=macd_long, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
    data['Volatility'] = data['Close'].rolling(window=rsi_window).std()
    data['Momentum'] = data['Close'] - data['Close'].shift(rsi_window)
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    return data

# Function to build the model
def build_model(hp, input_shape, architecture):
    model = Sequential()
    reg = l1_l2(l1=hp.Float('l1_reg', 1e-6, 0.01, sampling='log'), l2=hp.Float('l2_reg', 1e-6, 0.01, sampling='log'))

    if architecture == 'LSTM':
        num_layers = hp.Int('num_lstm_layers', 1, 2)
        for i in range(num_layers):
            model.add(Bidirectional(LSTM(units=hp.Int(f'units_lstm_{i}', 50, 150, step=50),
                                         return_sequences=(i < num_layers - 1),
                                         kernel_regularizer=reg),
                                    input_shape=input_shape))
            model.add(Dropout(hp.Float(f'dropout_lstm_{i}', 0.2, 0.4, step=0.1)))
    elif architecture == 'GRU':
        num_layers = hp.Int('num_gru_layers', 1, 2)
        for i in range(num_layers):
            model.add(Bidirectional(GRU(units=hp.Int(f'units_gru_{i}', 50, 150, step=50),
                                        return_sequences=(i < num_layers - 1),
                                        kernel_regularizer=reg),
                                    input_shape=input_shape))
            model.add(Dropout(hp.Float(f'dropout_gru_{i}', 0.2, 0.4, step=0.1)))
    elif architecture == 'CNN-LSTM':
        model.add(Conv1D(filters=hp.Int('filters', 32, 96, step=32),
                         kernel_size=hp.Int('kernel_size', 2, 4),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        num_layers = hp.Int('num_cnn_lstm_layers', 1, 2)
        for i in range(num_layers):
            model.add(Bidirectional(LSTM(units=hp.Int(f'units_cnn_lstm_{i}', 50, 150, step=50),
                                         return_sequences=(i < num_layers - 1),
                                         kernel_regularizer=reg)))
            model.add(Dropout(hp.Float(f'dropout_cnn_lstm_{i}', 0.2, 0.4, step=0.1)))

    model.add(Dense(1))
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')), loss='mean_squared_error')
    return model

# Class to manage stock prediction model
class StockPredictionModel:
    def __init__(self, symbol, n_steps, feature_columns, ema_window=14, rsi_window=14, macd_short=12, macd_long=26,
                 macd_signal=9, project_name='stock_prediction', architecture='LSTM'):
        self.symbol = symbol
        self.n_steps = n_steps
        self.n_days = 1
        self.feature_columns = feature_columns
        self.ema_window = ema_window
        self.rsi_window = rsi_window
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.macd_signal = macd_signal
        self.data = None
        self.models = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.project_name = project_name
        self.project_dir = f'my_dir/{project_name}'
        self.architecture = architecture
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)  # Clearing the project directory

    def fetch_data(self, period="10y"):
        logging.info(f'Fetching data for {self.symbol}')
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period)
        data.ffill(inplace=True)
        data = add_technical_indicators(data, self.ema_window, self.rsi_window, self.macd_short, self.macd_long, self.macd_signal)
        self.data = data[self.feature_columns].dropna().astype(float)

    def preprocess_data(self):
        logging.info('Preprocessing data')
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(self.n_steps, len(scaled_data)):
            X.append(scaled_data[i - self.n_steps:i])
            y.append(scaled_data[i, self.data.columns.get_loc('Close')])
        return np.array(X), np.array(y)

    def time_series_cross_validate(self, X, y, n_splits=3):
        logging.info('Starting time series cross-validation')
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_models = []
        tuner = BayesianOptimization(lambda hp: build_model(hp, (X.shape[1], X.shape[2]), self.architecture),
                                     objective=Objective("val_loss", direction="min"),
                                     max_trials=5,  # Reduced number of trials further
                                     executions_per_trial=1,
                                     directory=self.project_dir,
                                     project_name=self.project_name)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            tuner.search(X_train, y_train, epochs=30,  # Reduced number of epochs further
                         validation_data=(X_test, y_test),
                         callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])  # Reduced patience further
            best_models.append(tuner.get_best_models(num_models=1)[0])
        self.models = best_models
        return tuner

    def predict_future_prices(self, last_known_seq):
        logging.info('Predicting future prices')
        last_sequence_scaled = self.scaler.transform(last_known_seq).reshape(1, self.n_steps, len(self.feature_columns))
        ensemble_predictions = []
        for day in range(self.n_days):
            day_predictions = []
            for model in self.models:
                prediction_scaled = model.predict(last_sequence_scaled, verbose=0)[0][0]
                day_predictions.append(prediction_scaled)
            prediction_scaled_mean = np.mean(day_predictions)
            prediction = self.scaler.inverse_transform([[prediction_scaled_mean] + [0] * (len(self.feature_columns) - 1)])[0][0]
            ensemble_predictions.append(prediction)
            new_input = np.append(last_sequence_scaled[0, 1:], [[prediction_scaled_mean] + [0] * (len(self.feature_columns) - 1)], axis=0)
            last_sequence_scaled = new_input.reshape(1, self.n_steps, len(self.feature_columns))
        return ensemble_predictions

    def plot_predictions(self, historical_data, predicted_prices):
        current_date = datetime.now(pytz.timezone('America/New_York')).date()
        historical_dates = pd.date_range(end=current_date, periods=len(historical_data), freq='D')
        predicted_dates = pd.date_range(start=current_date + timedelta(days=1), periods=len(predicted_prices), freq='D')
        plt.figure(figsize=(10, 6))
        plt.plot(historical_dates, historical_data, label='Historical Closing Price')
        plt.plot(predicted_dates, predicted_prices, color='red', marker='o', linestyle='dashed', label='Predicted Closing Price')
        plt.title(f'Future Price Prediction for {self.symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def validate_accuracy(self, actual_price, predicted_price):
        return np.abs(actual_price - predicted_price) / actual_price

    def run(self):
        try:
            self.fetch_data()
            X, y = self.preprocess_data()
            self.time_series_cross_validate(X, y)
            last_known_seq = self.data[-self.n_steps:]
            predicted_prices = self.predict_future_prices(last_known_seq)
            last_actual_price = self.data['Close'].iloc[-1]
            last_predicted_price = predicted_prices[-1]
            accuracy = self.validate_accuracy(last_actual_price, last_predicted_price)
            logging.info(f'Validation Accuracy: {accuracy}')
            return accuracy
        except Exception as e:
            logging.error(f"Error in run method: {e}")
            return None

def experiment_with_combinations(symbol, n_step_options, window_sizes, feature_sets, architectures):
    results = Parallel(n_jobs=-1)(delayed(run_single_experiment)(symbol, n_steps, window_size, features, architecture)
                                   for n_steps in n_step_options
                                   for window_size in window_sizes
                                   for features in feature_sets
                                   for architecture in architectures)
    return results

def run_single_experiment(symbol, n_steps, window_size, features, architecture):
    logging.info(f"Testing with n_steps = {n_steps}, window size = {window_size}, features = {features}, architecture = {architecture}")
    unique_project_name = f'{symbol}_steps{n_steps}_window{window_size}_features{"_".join(features)}_arch{architecture}'
    model = StockPredictionModel(symbol, n_steps, features, window_size, window_size, window_size // 2,
                                 window_size, window_size // 3, unique_project_name, architecture)
    accuracy = model.run()
    if accuracy is not None:
        return (n_steps, window_size, features, architecture, accuracy)
    K.clear_session()
    gc.collect()
    return None

# Define different combinations of features
feature_sets = [
    ['Close', 'RSI', 'MACD'],
    ['Close', 'RSI', 'MACD', 'Volatility'],
    ['Close', 'RSI', 'MACD', 'EMA'],
    ['Close', 'RSI', 'MACD', 'Momentum'],
    ['Close', 'RSI', 'MACD', 'Log_Return'],
    ['Close', 'RSI', 'MACD', 'Volatility', 'Momentum', 'Log_Return', 'EMA'],
]

# Define different architectures to test
architectures = ['LSTM', 'GRU', 'CNN-LSTM']

# Initialize and run the combined experiments
symbol = 'AAPL'
n_step_options = [10]  # Example of different step sizes
window_sizes = [10, 20, 30, 50]  # Example of different window sizes for technical indicators
experiment_results = experiment_with_combinations(symbol, n_step_options, window_sizes, feature_sets, architectures)
logging.info(f"Experiment results: {experiment_results}")
