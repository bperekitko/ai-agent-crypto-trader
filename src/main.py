from data.raw_data import get_data, refresh_data
from model.features.analyze.analyze import analyze, log_transform, box_cox_transform, winsorize_transform
from model.features.close_price_prct_diff import CloseDiff
from sklearn.preprocessing import StandardScaler

from model.features.close_to_low import CloseToLow
from model.features.high_to_close import HighToClose
from model.features.close_to_sma20 import CloseToSma20
from model.softmax_regression.softmax import SoftmaxRegression


def main():
    # refresh_data()
    df = get_data()
    SoftmaxRegression(df).describe()
    # feature = CloseToSma20()
    # values = feature.calculate(df)
    # df['high'] = values
    # df = df.dropna()
    # analyze(df['high'], f'{feature.name()}_log_transformed')
    # df['high'] = log_transform(values)
    # analyze(log_transformed, f'{feature.name()}_log_transformed')
    #
    # box_coxed, lambda_fitted = box_cox_transform(df['high'])
    # df['high'] = box_coxed
    # analyze(df['boxed'], f'{feature.name()}_boxed_coxed')
    # df['high'], min_limit, max_limit = winsorize_transform(df['high'])
    # print(f'Winsor min: ', min_limit)
    # print(f'Winsor max: ', max_limit)
    #
    # scaler = StandardScaler()
    # df['high'] = scaler.fit_transform(df[['high']])
    # analyze(df['high'], f'{feature.name()}')


if __name__ == "__main__":
    main()
