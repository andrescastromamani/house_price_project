from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras
import typer

from house_price.config import AUTOENCODER_MODEL_PATH, MLP_MODEL_PATH, PROCESSED_DATA_DIR
from house_price.modeling.train import AutoencoderModel, MLPModel

app = typer.Typer()

THRESHOLD = 0.5


class ModelEvaluator:
    """Computes classification metrics and PR curves for a set of models."""

    def __init__(self, y_test: np.ndarray) -> None:
        self.y_test = y_test
        self.results: list[dict] = []

    def evaluate(self, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
        """Evaluate each prediction and store a summary row per model."""
        self.results = []
        for name, probs in predictions.items():
            preds = (probs >= THRESHOLD).astype(int)
            self.results.append(
                {
                    "Modelo": name,
                    "Accuracy": accuracy_score(self.y_test, preds),
                    "Precision": precision_score(self.y_test, preds, zero_division=0),
                    "Recall": recall_score(self.y_test, preds, zero_division=0),
                    "F1-Score": f1_score(self.y_test, preds, zero_division=0),
                    "ROC-AUC": roc_auc_score(self.y_test, probs),
                    "PR-AUC": average_precision_score(self.y_test, probs),
                }
            )
        return pd.DataFrame(self.results)

    def plot_precision_recall(
        self,
        predictions: dict[str, np.ndarray],
        output_path: Path | None = None,
    ) -> None:
        """Plot the Precision-Recall curves for every model."""
        plt.figure(figsize=(8, 6))
        for name, probs in predictions.items():
            precision, recall, _ = precision_recall_curve(self.y_test, probs)
            pr_auc = average_precision_score(self.y_test, probs)
            plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Comparación de Curvas Precision-Recall (PR-AUC)")
        plt.legend()
        if output_path is not None:
            plt.savefig(output_path, bbox_inches="tight")
            logger.info(f"PR curve saved to {output_path}")
        else:
            plt.show()


def run_inference(
    X_test: np.ndarray,
    mlp_path: Path = MLP_MODEL_PATH,
    autoencoder_path: Path = AUTOENCODER_MODEL_PATH,
) -> dict[str, np.ndarray]:
    """Load trained models and return probability scores for the test set."""
    mlp_model = keras.models.load_model(mlp_path)
    autoencoder_model = keras.models.load_model(autoencoder_path)

    mlp_wrapper = MLPModel(X_test.shape[1])
    mlp_wrapper.model = mlp_model
    autoencoder_wrapper = AutoencoderModel(X_test.shape[1])
    autoencoder_wrapper.model = autoencoder_model

    return {
        "DL 1: MLP Supervisado": mlp_wrapper.predict(X_test).ravel(),
        "DL 2: Autoencoder": autoencoder_wrapper.reconstruction_error(X_test),
    }


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    mlp_model_path: Path = MLP_MODEL_PATH,
    autoencoder_model_path: Path = AUTOENCODER_MODEL_PATH,
    output_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
) -> None:
    X_test = pd.read_csv(features_path).to_numpy()
    y_test = pd.read_csv(labels_path).to_numpy().ravel()

    predictions = run_inference(
        X_test,
        mlp_path=mlp_model_path,
        autoencoder_path=autoencoder_model_path,
    )
    evaluator = ModelEvaluator(y_test)
    summary = evaluator.evaluate(predictions)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    app()
