from data.raw_data import get_data
from model.features.close_price_prct_diff import CloseDiff

from model.softmax_regression.softmax import SoftmaxRegression


def main():
    # refresh_data()
    df = get_data()
    train_size = int(len(df) * 0.75)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    model = SoftmaxRegression()
    model.train(train_data)
    model.test(test_data)
    # model.test_different_lags(train_data, test_data)


if __name__ == "__main__":
    main()
