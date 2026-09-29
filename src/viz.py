import matplotlib.pyplot as plt
import numpy as np
from src.utils import save_output
from src.config import SMOKER_PALETTE


# 1. DISTRIBUCION DE CHARGES

def plot_hist_charges(df, save=False, filename="hist_charges.png"):

    mean_charges = df["charges"].mean()
    median_charges = df["charges"].median()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(df["charges"], bins=30, color="#2C5F7C", edgecolor="white")
    ax.axvline(mean_charges, color="#D9822B", linestyle="--", linewidth=1.5,
               label=f"Media: ${mean_charges:,.0f}")
    ax.axvline(median_charges, color="#3C8C5C", linestyle="--", linewidth=1.5,
               label=f"Mediana: ${median_charges:,.0f}")

    ax.set_xlabel("Gasto medico anual ($)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion de charges (gasto medico anual)")
    ax.legend()
    ax.grid(axis="y", color="#dddddd")

    plt.tight_layout()

    if save:
        save_output(fig, filename)
    else:
        plt.show()


# 2. BOXPLOT CHARGES POR FUMADOR

def plot_boxplot_smoker(df, save=False, filename="boxplot_smoker.png"):

    fig, ax = plt.subplots(figsize=(9, 5))

    data = [df[df["smoker"] == "no"]["charges"], df[df["smoker"] == "yes"]["charges"]]

    bp = ax.boxplot(data, tick_labels=["No fumador", "Fumador"], patch_artist=True)

    for patch, color in zip(bp["boxes"], [SMOKER_PALETTE["no"], SMOKER_PALETTE["yes"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Gasto medico anual ($)")
    ax.set_title("Charges por condicion de fumador")
    ax.grid(axis="y", color="#dddddd")

    plt.tight_layout()

    if save:
        save_output(fig, filename)
    else:
        plt.show()


# 3. REGRESION SIMPLE: CHARGES ~ BMI

def plot_regression_simple(df, model, r2, save=False, filename="regression_bmi.png"):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = df["smoker"].map(SMOKER_PALETTE)
    ax.scatter(df["bmi"], df["charges"], c=colors, alpha=0.5, s=15)

    bmi_range = np.linspace(df["bmi"].min(), df["bmi"].max(), 100).reshape(-1, 1)
    pred_line = model.predict(bmi_range)
    ax.plot(bmi_range, pred_line, color="black", linewidth=2)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=SMOKER_PALETTE["no"], label="No fumador"),
        Patch(facecolor=SMOKER_PALETTE["yes"], label="Fumador"),
        plt.Line2D([0], [0], color="black", linewidth=2, label=f"Recta OLS (R2={r2:.2f})"),
    ]
    ax.legend(handles=legend_elems, loc="upper left")

    ax.set_xlabel("BMI (indice de masa corporal)")
    ax.set_ylabel("Gasto medico anual ($)")
    ax.set_title("Regresion lineal simple: charges ~ bmi")
    ax.grid(color="#dddddd")

    plt.tight_layout()

    if save:
        save_output(fig, filename)
    else:
        plt.show()


# 4. RESIDUOS DEL MODELO MULTIPLE

def plot_residuals(predictions, residuals, save=False, filename="residuals.png"):

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(predictions, residuals, alpha=0.4, s=15, color="#2C5F7C")
    ax.axhline(0, color="#D9822B", linewidth=1.5)

    ax.set_xlabel("Valores predichos ($)")
    ax.set_ylabel("Residuos ($)")
    ax.set_title("Grafico de residuos - Modelo multiple")
    ax.grid(color="#dddddd")

    plt.tight_layout()

    if save:
        save_output(fig, filename)
    else:
        plt.show()
