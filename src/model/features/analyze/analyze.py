import matplotlib.pyplot as plt
import numpy as np

from model.features.analyze import current_dir_path_for
from scipy.stats import boxcox

from utils.log import get_logger


def analyze(values, feature_name):
    __histogram(values, feature_name)
    __box_plot(values, feature_name)
    __describe(feature_name, values)


def log_transform(values):
    shift_value = abs(values.min()) + 1
    return np.log1p(values + shift_value)


def box_cox_transform(values, lambda_fit=None, shift_value=None):
    to_shift = abs(values.min()) + 1 if shift_value is None else shift_value
    if lambda_fit is None:
        transformed_data, fitted_lambda = boxcox(values + to_shift)
        return transformed_data, fitted_lambda, to_shift
    else:
        return boxcox(values + to_shift, lambda_fit)


def winsorize_transform(values, top=99, bottom=1):
    max_limit = np.percentile(values, top)
    min_limit = np.percentile(values, bottom)
    return np.clip(values, a_min=min_limit, a_max=max_limit), min_limit, max_limit


def __describe(feature_name, values):
    # Podstawowe statystyki opisowe
    mean = values.mean()
    median = values.median()
    std_dev = values.std()
    variance = values.var()
    # Kwartyle i IQR
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    # Miary asymetrii i koncentracji
    skewness = values.skew()
    kurtosis = values.kurt()

    logger = get_logger(feature_name)

    # Wyświetlanie wyników

    logger.info(f"[{feature_name}] Mean: {mean}")
    logger.info(f"[{feature_name}] Median: {median}")
    logger.info(f"[{feature_name}] Standard deviation: {std_dev}")
    logger.info(f"[{feature_name}] Variance: {variance}")
    logger.info(f"[{feature_name}] First quantile (Q1): {q1}")
    logger.info(f"[{feature_name}] Third quantile (Q3): {q3}")
    logger.info(f"[{feature_name}] IQR: {iqr}")
    logger.info(f"[{feature_name}] Skewness: {skewness}")
    logger.info(f"[{feature_name}] kurtosis: {kurtosis}")


def __box_plot(values, name):
    # Wykres pudełkowy (box plot) do identyfikacji wartości odstających
    plt.figure(figsize=(10, 6))
    plt.boxplot(values, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'{name} BoxPlot')
    plt.xlabel("Values")
    plt.grid(True)
    # Zapisz wykres pudełkowy do pliku
    plt.savefig(current_dir_path_for(f'{name}') + f'/{name}_boxplot.png')
    plt.close()


def __histogram(values, name):
    # Histogram rozkładu procentowej zmiany
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, color='blue', edgecolor='black')
    plt.title(f"{name} histogram")
    plt.xlabel("Values")
    plt.ylabel("Count")
    plt.grid(True)
    # Zapisz histogram do pliku
    plt.savefig(current_dir_path_for(f'{name}') + f'/{name}_histogram.png')
    plt.close()
