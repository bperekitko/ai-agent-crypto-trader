from data.raw_data import get_data, refresh_data
from model.features.analyze.analyze import analyze, log_transform, box_cox_transform, winsorize_transform
from model.features.close_price_prct_diff import CloseDiff
from sklearn.preprocessing import StandardScaler

from model.features.high_to_close import HighToClose
from model.softmax_regression.softmax import SoftmaxRegression


def main():
    # refresh_data()
    df = get_data()
    # SoftmaxRegression(df).describe()
    feature = HighToClose()
    values = feature.calculate(df)
    df['boxed'] = values
    df = df.dropna()
    # log_transformed = log_transform(values)
    # analyze(log_transformed, f'{feature.name()}_log_transformed')

    # box_coxed, lambda_fitted = box_cox_transform(df['boxed'])
    # df['boxed'] = box_coxed
    # analyze(df['boxed'], f'{feature.name()}_boxed_coxed')
    df['boxed'], min_limit, max_limit = winsorize_transform(df['boxed'])
    print(f'Winsor min: ', min_limit)
    print(f'Winsor max: ', max_limit)

    scaler = StandardScaler()
    df['boxed'] = scaler.fit_transform(df[['boxed']])
    analyze(df['boxed'], f'{feature.name()}_winsorized')


if __name__ == "__main__":
    main()
