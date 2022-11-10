import sentiment_data_collection
import sentiment_analysis


def main_run():
    return sentiment_analysis.sentiment_model(sentiment_data_collection.retrieve_requested_data())


if __name__ == "__main__":
    print(main_run())