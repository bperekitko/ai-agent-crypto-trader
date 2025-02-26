import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from model.features.analyze import current_dir_path_for


def analyze_importance(features_data: pd.DataFrame, target_data: pd.Series):
    x = features_data.copy()
    y = target_data.values.ravel()
    feature_cols = x.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(x_train, y_train)

    importances = rf_model.feature_importances_
    feat_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    feat_importance_df.sort_values(by='importance', ascending=False, inplace=True)

    print("Feature Importances (RandomForest):")
    print(feat_importance_df)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_importance_df['feature'], feat_importance_df['importance'])
    plt.xlabel("Feature importance")
    plt.title("Feature importance - RandomForest")
    plt.gca().invert_yaxis()
    plt.savefig(current_dir_path_for('importance') + f'/importance.png')
    plt.close()

    explainer = shap.Explainer(rf_model, x_train)
    shap_values = explainer(x_test, check_additivity=False)
    shap_values_class_1 = shap_values.values[..., 1]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_class_1, x_test, plot_type="bar", feature_names=np.array(feature_cols), show=False)
    plt.tight_layout()
    plt.savefig(current_dir_path_for('importance') + "/shap_feature_importance_bar.png")
    plt.close()

    shap.summary_plot(shap_values_class_1, x_test, feature_names=np.array(feature_cols), show=False)
    plt.tight_layout()
    plt.savefig(current_dir_path_for('importance') + "/shap_summary_plot.png")
    plt.close()
