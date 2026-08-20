from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import typer

from house_price.config import PROCESSED_DATA_DIR

app = typer.Typer()


class FeatureEngineer:
    """Splits the dataset and scales numeric features without data leakage."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self._amount_scaler = RobustScaler()
        self._time_scaler = RobustScaler()
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    def prepare(
        self,
        frame: pd.DataFrame,
        target: str = "Class",
    ) -> None:
        """Split the frame and scale Time/Amount using scalers fit on train only."""
        features = frame.drop(target, axis=1)
        labels = frame[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        self._amount_scaler.fit(self.X_train["Amount"].values.reshape(-1, 1))
        self._time_scaler.fit(self.X_train["Time"].values.reshape(-1, 1))
        self.X_train = self._scale(self.X_train)
        self.X_test = self._scale(self.X_test)
        logger.info(f"Train {self.X_train.shape} - Test {self.X_test.shape} after scaling")

    def class_weights(self) -> dict:
        """Compute per-class weights to counter the class imbalance."""
        if self.y_train is None:
            raise RuntimeError("Call prepare() before computing class weights.")
        neg_count, pos_count = np.bincount(self.y_train)
        n_samples = len(self.y_train)
        return {
            0: (1 / neg_count) * (n_samples / 2.0),
            1: (1 / pos_count) * (n_samples / 2.0),
        }

    def _scale(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace Time/Amount with their robust-scaled versions."""
        scaled = frame.copy()
        scaled["scaled_amount"] = self._amount_scaler.transform(
            scaled["Amount"].values.reshape(-1, 1)
        )
        scaled["scaled_time"] = self._time_scaler.transform(scaled["Time"].values.reshape(-1, 1))
        return scaled.drop(["Time", "Amount"], axis=1)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> None:
    # ------------------------------------------------------------------
    # This entry point requires a preprocessed dataset file. The OOP
    # pipeline (FeatureEngineer.prepare) is the canonical way to use it.
    # ------------------------------------------------------------------
    logger.info("Generating features from dataset...")
    frame = pd.read_csv(input_path)
    engineer = FeatureEngineer()
    engineer.prepare(frame)
    pd.concat([engineer.X_train, engineer.y_train], axis=1).to_csv(output_path, index=False)
    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
