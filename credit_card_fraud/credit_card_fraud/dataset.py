from pathlib import Path

from loguru import logger
import pandas as pd
# pyrefly: ignore [missing-import]
import typer

# pyrefly: ignore [missing-import]
from credit_card_fraud.config import RAW_DATA_DIR

app = typer.Typer()


class CreditCardDataset:
    """Loads the credit card transactions dataset from disk."""

    def __init__(self, data_path: Path = RAW_DATA_DIR / "creditcard.csv") -> None:
        self.data_path = data_path
        self.frame: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Read the CSV file and keep it in the frame attribute."""
        logger.info(f"Loading dataset from {self.data_path}")
        self.frame = pd.read_csv(self.data_path)
        logger.success(f"Dataset loaded with shape {self.frame.shape}")
        return self.frame


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "creditcard.csv",
) -> None:
    dataset = CreditCardDataset(data_path=input_path)
    dataset.load()


if __name__ == "__main__":
    app()
