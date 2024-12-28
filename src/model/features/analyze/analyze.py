import matplotlib.pyplot as plt
import numpy as np

from model.features.analyze import current_dir_path_for
from scipy.stats import boxcox


def analyze(values, feature_name):
    __histogram(values, feature_name)
    __box_plot(values, feature_name)
    __describe(feature_name, values)


def log_transform(values):
    shift_value = abs(values.min()) + 1
    return np.log1p(values + shift_value)


def box_cox_transform(values):
    shift_value = abs(values.min()) + 1
    return boxcox(values + shift_value)


def winsorize_transform(values):
    max_limit = np.percentile(values, 99)
    min_limit = np.percentile(values, 1)
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

    # Wyświetlanie wyników
    print(f'\n\n {feature_name}: \n')
    print(f"[{feature_name}] Mean: {mean}")
    print(f"[{feature_name}] Median: {median}")
    print(f"[{feature_name}] Standard deviation: {std_dev}")
    print(f"[{feature_name}] Variance: {variance}")
    print(f"[{feature_name}] First quantile (Q1): {q1}")
    print(f"[{feature_name}] Third quantile (Q3): {q3}")
    print(f"[{feature_name}] IQR: {iqr}")
    print(f"[{feature_name}] Skewness: {skewness}")
    print(f"[{feature_name}] kurtosis: {kurtosis}")


def __box_plot(values, name):
    # Wykres pudełkowy (box plot) do identyfikacji wartości odstających
    plt.figure(figsize=(10, 6))
    plt.boxplot(values, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'{name} BoxPlot')
    plt.xlabel("Values")
    plt.grid(True)
    # Zapisz wykres pudełkowy do pliku
    plt.savefig(current_dir_path_for(f'{name}_boxplot.png'))
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
    plt.savefig(current_dir_path_for(f'{name}_histogram.png'))
    plt.close()
