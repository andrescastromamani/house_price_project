from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from house_price.config import AUTOENCODER_MODEL_PATH, MLP_MODEL_PATH

app = typer.Typer()


class BaseModel(ABC):
    """Common interface for the deep learning models used in the pipeline."""

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.model: keras.Model | None = None
        self.history: keras.callbacks.History | None = None

    @abstractmethod
    def build(self) -> "BaseModel":
        """Construct and compile the underlying Keras model."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        checkpoint_path: Path = Path(""),
        epochs: int = 30,
        batch_size: int = 2048,
    ) -> keras.callbacks.History:
        """Train the model and return the training history."""

    def save(self, checkpoint_path: Path) -> None:
        """Serialize the model to disk."""
        if self.model is None:
            raise RuntimeError("Model not built yet.")
        self.model.save(checkpoint_path)
        logger.success(f"Model saved to {checkpoint_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the raw model output for the given features."""
        if self.model is None:
            raise RuntimeError("Model not built yet.")
        return self.model.predict(X, batch_size=2048)


class MLPModel(BaseModel):
    """Supervised multi-layer perceptron for binary fraud classification."""

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__(input_dim)
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

    def build(self) -> "MLPModel":
        self.model = keras.Sequential(
            [
                layers.Input(shape=(self.input_dim,)),
                layers.Dense(64, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(32, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="MLP_Supervisado",
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"), "accuracy"],
        )
        logger.info("MLP built and compiled.")
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        checkpoint_path: Path = MLP_MODEL_PATH,
        epochs: int = 30,
        batch_size: int = 2048,
        validation_split: float = 0.15,
        class_weight: dict | None = None,
    ) -> keras.callbacks.History:
        if self.model is None:
            raise RuntimeError("Call build() before training.")
        callbacks = [
            EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True
            ),
            ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_pr_auc",
                mode="max",
                save_best_only=True,
            ),
        ]
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        return self.history


class AutoencoderModel(BaseModel):
    """Autoencoder that reconstructs normal transactions to flag anomalies."""

    def __init__(self, input_dim: int, encoding_dim: int = 14) -> None:
        super().__init__(input_dim)
        self.encoding_dim = encoding_dim

    def build(self) -> "AutoencoderModel":
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(20, activation="tanh")(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation="relu")(encoded)
        decoded = layers.Dense(20, activation="tanh")(encoded)
        decoded = layers.Dense(self.input_dim, activation="linear")(decoded)

        self.model = keras.Model(
            inputs=input_layer, outputs=decoded, name="Autoencoder_Anomalias"
        )
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        logger.info("Autoencoder built and compiled.")
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        checkpoint_path: Path = AUTOENCODER_MODEL_PATH,
        epochs: int = 30,
        batch_size: int = 2048,
        validation_split: float = 0.15,
    ) -> keras.callbacks.History:
        if self.model is None:
            raise RuntimeError("Call build() before training.")
        callbacks = [
            EarlyStopping(
                monitor="val_loss", mode="min", patience=5, restore_best_weights=True
            ),
            ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            ),
        ]
        self.history = self.model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        return self.history

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return the normalized MSE reconstruction error as anomaly score."""
        reconstructed = self.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        mse_min = mse.min()
        mse_max = mse.max()
        return (mse - mse_min) / (mse_max - mse_min)


@app.command()
def main(
    features_path: Path = typer.Option(
        "data/processed/features.csv", help="Processed feature matrix"
    ),
    labels_path: Path = typer.Option(
        "data/processed/labels.csv", help="Target labels"
    ),
    model_path: Path = typer.Option(str(MLP_MODEL_PATH), help="Output model path"),
) -> None:
    logger.info("Training some model...")
    logger.info(f"Features {features_path} - Labels {labels_path}")
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()