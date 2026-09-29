from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def fit_simple_regression(df, x_col="bmi", y_col="charges"):
    X = df[[x_col]]
    y = df[y_col]

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)

    return model, predictions, r2


def fit_multiple_regression(df, feature_cols, y_col="charges"):
    X = df[feature_cols]
    y = df[y_col]

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    residuals = y - predictions

    return model, predictions, residuals, r2
