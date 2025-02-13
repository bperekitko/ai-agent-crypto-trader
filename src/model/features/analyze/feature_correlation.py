import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model.features.analyze import current_dir_path_for


def analyze_correlation(df: pd.DataFrame):
    corr_matrix = df.corr()
    plt.figure(figsize=(20,16))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation matrix')
    plt.savefig(current_dir_path_for('correlation') + f'/correlation.png')
    plt.close()