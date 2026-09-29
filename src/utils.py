import os
import matplotlib.pyplot as plt


def save_output(fig, filename, folder="output"):

    # 1. Asegurar que la carpeta existe
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Carpeta '{folder}' creada.")

    # 2. Construir la ruta completa
    path = os.path.join(folder, filename)

    # 3. Guardar figura
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Grafico guardado en: {path}")

    # 4. Liberar memoria
    plt.close(fig)


def export_csv(df, filename, folder="data/processed"):

    # 1. Asegurar que la carpeta existe
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Carpeta '{folder}' creada.")

    # 2. Construir la ruta completa
    path = os.path.join(folder, filename)

    # 3. Exportar CSV
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"Datos exportados a: {path}")
