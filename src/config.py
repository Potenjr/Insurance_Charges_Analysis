# Rutas
DATA_PATH = "data/raw/insurance.csv"

# Umbrales
IQR_FACTOR = 1.5
ZSCORE_THRESHOLD = 3

# Variable objetivo y predictores
TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]

# Paleta de colores
SMOKER_PALETTE = {"no": "#2C5F7C", "yes": "#D9822B"}
