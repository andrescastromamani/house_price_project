from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from credit_card_fraud.config import FIGURES_DIR, RAW_DATA_DIR
from credit_card_fraud.dataset import CreditCardDataset

app = typer.Typer()


class ExploratoryPlots:
    """Generates the EDA visualizations for the transactions dataset."""

    def __init__(self, output_dir: Path = FIGURES_DIR) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def class_distribution(self, frame: pd.DataFrame) -> None:
        plt.figure(figsize=(7, 4))
        sns.countplot(
            x="Class",
            data=frame,
            hue="Class",
            palette=["#67db67", "#d9534f"],
            legend=False,
        )
        plt.yscale("log")
        plt.title("Distribución de Clases")
        plt.xlabel("Clase (0: No Fraude, 1: Fraude)")
        plt.ylabel("Cantidad de Transacciones")
        self._save("class_distribution.png")

    def correlation_heatmap(self, frame: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 10))
        sns.heatmap(frame.corr(), square=True)
        plt.title("Matriz de Correlación")
        self._save("correlation_heatmap.png")

    def amounts_by_class(self, frame: pd.DataFrame) -> None:
        fraud_frame = frame[frame["Class"] == 1]
        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.ecdfplot(
            data=frame[frame["Amount"] < 2200],
            x="Amount",
            hue="Class",
            ax=axes[0],
        )
        sns.boxplot(data=fraud_frame, y="Amount", ax=axes[1])
        axes[0].set_title("Distribución del Monto por Clase")
        axes[1].set_title("Monto de Transacciones Fraudulentas")
        plt.tight_layout()
        self._save("amounts_by_class.png")

    def time_distribution(self, frame: pd.DataFrame) -> None:
        fraud_frame = frame[frame["Class"] == 1]
        _, axes = plt.subplots(1, 2, figsize=(16, 5))
        sns.histplot(data=frame, x="Time", bins=24, color="lightblue", ax=axes[0])
        sns.histplot(data=fraud_frame, x="Time", bins=24, color="red", ax=axes[1])
        axes[0].set_title("Distribución de Transacciones en el Tiempo")
        axes[1].set_title("Distribución de Fraudes en el Tiempo")
        plt.suptitle("Distribution of transactions and fraudulent transactions over time")
        self._save("time_distribution.png")

    def _save(self, filename: str) -> None:
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches="tight")
        plt.show()
        logger.info(f"Saved {output_path}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "creditcard.csv",
    output_dir: Path = FIGURES_DIR,
) -> None:
    dataset = CreditCardDataset(data_path=input_path)
    frame = dataset.load()

    plots = ExploratoryPlots(output_dir=output_dir)
    plots.class_distribution(frame)
    plots.correlation_heatmap(frame)
    plots.amounts_by_class(frame)
    plots.time_distribution(frame)


if __name__ == "__main__":
    app()
