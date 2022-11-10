import LSTM_model

def main_run():
    data = LSTM_model.get_data(input("Provide Stock Ticker: "))
    x_train, y_train, x_test, y_test = LSTM_model.clean_data(data)
    model = LSTM_model.setup_model(x_train, y_train)
    return LSTM_model.predicts(model)

if __name__ == "__main__":
    print(main_run())