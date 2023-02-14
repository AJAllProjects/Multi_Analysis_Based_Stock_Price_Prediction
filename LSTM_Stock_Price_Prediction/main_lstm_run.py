import LSTM_Stock_Price_Prediction.LSTM_model as LSTM_model

def main_run(ticker = "TSLA"):
    stock_data = LSTM_model.get_data(ticker)
    x_train, x_test, y_train, y_test, training_data_len, scaler, scaled_data = LSTM_model.clean_split_data(stock_data)
    predictions, model = LSTM_model.train_model_predict(x_train, y_train, x_test, scaler)
    # LSTM_model.show_predictions(stock_data, training_data_len, predictions)
    if LSTM_model.predict_next_day(model, scaled_data, scaler) > predictions[len(predictions) - 1]:
        return 1
    else:
        return 0

if __name__ == "__main__":
    print(main_run())









