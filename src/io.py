import pandas as pd
from src.config import DATA_PATH


def load_data():
    return pd.read_csv(DATA_PATH)


def save_results(df, name):
    df.to_csv(f"output/{name}.csv", index=False)
