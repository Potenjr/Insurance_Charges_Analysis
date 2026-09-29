def clean_insurance_data(df):
    df_clean = df.copy()

    # Eliminar duplicados exactos si los hubiera
    df_clean = df_clean.drop_duplicates()

    # Eliminar filas con valores nulos en columnas clave
    df_clean = df_clean.dropna(subset=["age", "bmi", "children", "charges"])

    # Normalizar texto en columnas categoricas
    for col in ["sex", "smoker", "region"]:
        df_clean[col] = df_clean[col].str.strip().str.lower()

    return df_clean
