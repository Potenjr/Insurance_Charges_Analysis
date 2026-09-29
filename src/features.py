import numpy as np
from src.config import IQR_FACTOR, ZSCORE_THRESHOLD


def generate_dummy_variables(df):
    df_feat = df.copy()
    df_feat["smoker_yes"] = (df_feat["smoker"] == "yes").astype(int)
    df_feat["sex_male"] = (df_feat["sex"] == "male").astype(int)
    return df_feat


def detect_outliers_iqr(df, column="charges", factor=IQR_FACTOR):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - factor * IQR
    lim_sup = Q3 + factor * IQR
    mask = (df[column] < lim_inf) | (df[column] > lim_sup)
    return mask, lim_inf, lim_sup


def detect_outliers_zscore(df, column="charges", threshold=ZSCORE_THRESHOLD):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = (df[column] - mean) / std
    mask = np.abs(z_scores) > threshold
    return mask
