import pandas as pd
from scipy import stats

from src.io import load_data
from src.cleaning import clean_insurance_data
from src.features import generate_dummy_variables, detect_outliers_iqr, detect_outliers_zscore
from src.model import fit_simple_regression, fit_multiple_regression
from src.viz import (
    plot_hist_charges,
    plot_boxplot_smoker,
    plot_regression_simple,
    plot_residuals,
)
from src.utils import export_csv


def main():

    # 1. CARGA DE DATOS

    print("Cargando datos...")
    df = load_data()

    # 2. LIMPIEZA

    print("Limpiando datos...")
    df = clean_insurance_data(df)

    # 3. FEATURE ENGINEERING

    print("Generando variables dummy...")
    df = generate_dummy_variables(df)

    # 4. DATASET PROCESADO

    print("Exportando dataset procesado...")
    export_csv(df, "insurance_processed.csv")

    # INSIGHT 1: ESTADISTICA DESCRIPTIVA DE CHARGES

    print("Calculando estadistica descriptiva de charges...")

    mean_charges = df["charges"].mean()
    median_charges = df["charges"].median()
    std_charges = df["charges"].std()
    cv_charges = (std_charges / mean_charges) * 100
    skew_charges = stats.skew(df["charges"])
    kurt_charges = stats.kurtosis(df["charges"])

    descriptive_summary = pd.DataFrame([{
        "mean": mean_charges,
        "median": median_charges,
        "std": std_charges,
        "cv_pct": cv_charges,
        "skewness": skew_charges,
        "kurtosis": kurt_charges,
    }])
    export_csv(descriptive_summary, "descriptive_summary.csv")

    smoker_summary = (
        df.groupby("smoker")["charges"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    export_csv(smoker_summary, "smoker_summary.csv")

    plot_hist_charges(df, save=True, filename="hist_charges.png")

    # INSIGHT 2: DETECCION DE OUTLIERS

    print("Detectando outliers...")

    mask_iqr, lim_inf, lim_sup = detect_outliers_iqr(df, column="charges")
    mask_z = detect_outliers_zscore(df, column="charges")

    outliers_summary = pd.DataFrame([{
        "limite_inferior_iqr": lim_inf,
        "limite_superior_iqr": lim_sup,
        "n_outliers_iqr": int(mask_iqr.sum()),
        "pct_outliers_iqr": mask_iqr.mean() * 100,
        "n_outliers_zscore": int(mask_z.sum()),
        "pct_fumadores_en_outliers_iqr": (df.loc[mask_iqr, "smoker"] == "yes").mean() * 100,
    }])
    export_csv(outliers_summary, "outliers_summary.csv")

    plot_boxplot_smoker(df, save=True, filename="boxplot_smoker.png")

    # INSIGHT 3: REGRESION LINEAL SIMPLE (CHARGES ~ BMI)

    print("Ajustando regresion lineal simple (charges ~ bmi)...")

    model_simple, pred_simple, r2_simple = fit_simple_regression(df, x_col="bmi", y_col="charges")

    simple_summary = pd.DataFrame([{
        "intercepto": model_simple.intercept_,
        "coef_bmi": model_simple.coef_[0],
        "r2": r2_simple,
    }])
    export_csv(simple_summary, "regression_simple_summary.csv")

    plot_regression_simple(df, model_simple, r2_simple, save=True, filename="regression_bmi.png")

    # INSIGHT 4: REGRESION LINEAL MULTIPLE

    print("Ajustando regresion lineal multiple...")

    features = ["age", "bmi", "children", "smoker_yes"]
    model_multi, pred_multi, residuals_multi, r2_multi = fit_multiple_regression(
        df, feature_cols=features, y_col="charges"
    )

    multi_summary = pd.DataFrame([{
        "intercepto": model_multi.intercept_,
        **{f"coef_{f}": c for f, c in zip(features, model_multi.coef_)},
        "r2": r2_multi,
    }])
    export_csv(multi_summary, "regression_multiple_summary.csv")

    correlation_matrix = df[["age", "bmi", "children"]].corr()
    export_csv(correlation_matrix.reset_index(), "correlation_matrix.csv")

    plot_residuals(pred_multi, residuals_multi, save=True, filename="residuals.png")

    # FIN

    print("\nPROCESO COMPLETADO")
    print(f"R2 modelo simple (bmi): {r2_simple:.4f}")
    print(f"R2 modelo multiple (age+bmi+children+smoker): {r2_multi:.4f}")
    print("CSV procesados en: data/processed/")
    print("Graficos en: output/")


if __name__ == "__main__":
    main()
