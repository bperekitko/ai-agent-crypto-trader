from data.raw_data import get_data
from model.softmax_regression.softmax import SoftmaxRegression


def main():
    # refresh_data()
    df = get_data()
    train_size = int(len(df) * 0.75)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    model = SoftmaxRegression()
    # for i in range(1, 11):
    #     model.test_volume_lagged(i, train_data, test_data)
    model.train(train_data)
    model.test(test_data)


if __name__ == "__main__":
    main()
