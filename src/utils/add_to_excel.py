import pandas as pd
from pandas import DataFrame


def append_df_to_excel(df: DataFrame, file_path: str):
    try:
        existing_excel = pd.read_excel(file_path)
        combined_data = pd.concat([existing_excel, df], ignore_index=True)
        combined_data.to_excel(file_path, index=False)
    except FileNotFoundError:
        df.to_excel(file_path, index=False)
