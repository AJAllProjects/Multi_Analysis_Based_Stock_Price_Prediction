import Decision_Tree_Trade_Bot.decision_tree_generation as bot
import LSTM_Stock_Price_Prediction.LSTM_model as lstm
import Ticker_Sentiment_Analysis.main_run as sentiment_analysis
import Volatility_Check.volatility_prediction as volatility_predict
from multiprocessing import Process

# Currently only running on 1 Stock Ticker (TSLA)
def multi_step_run():
    #Testing Multiprocessing/Multithreading capability to reduce operational times
        # trade_bot = Process(target=bot.make_buy_decision())
        # trade_bot.start()
        #
        # lstm_price = Process(target=lstm.main_run())
        # lstm_price.start()
        #
        # sentiment = Process(target=sentiment_analysis.main_run())
        # sentiment.start()
        #
        # volatility = Process(target=volatility_predict.volatility_predict())
        # volatility.start()
        #
        # trade_bot.join()
        # lstm_price.join()
        # sentiment.join()
        # volatility.join()

    print(bot.make_buy_decision())
    print(lstm.main_run())
    print(sentiment_analysis.main_run())
    print(volatility_predict.volatility_predict())

    # print(trade_bot)
    # print(lstm_price)
    # print(sentiment)
    # print(volatility)



if __name__ == "__main__":
    multi_step_run()

