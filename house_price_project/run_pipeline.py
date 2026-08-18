"""End-to-end pipeline: load data, engineer features, train and evaluate models.

Usage:
    python run_pipeline.py
"""

from house_price.config import REPORTS_DIR
from house_price.dataset import CreditCardDataset
from house_price.features import FeatureEngineer
from house_price.modeling.predict import ModelEvaluator
from house_price.modeling.train import AutoencoderModel, MLPModel

EPOCHS = 30
BATCH_SIZE = 2048
SEED = 42


def main() -> None:
    """Run the full fraud-detection pipeline."""
    dataset = CreditCardDataset()
    frame = dataset.load()

    engineer = FeatureEngineer(random_state=SEED)
    engineer.prepare(frame)
    class_weights = engineer.class_weights()
    print(f"X_train {engineer.X_train.shape} - X_test {engineer.X_test.shape}")
    print(f"Class weights: {class_weights}")

    mlp = MLPModel(input_dim=engineer.X_train.shape[1])
    mlp.build()
    mlp.train(
        engineer.X_train.to_numpy(),
        engineer.y_train.to_numpy(),
        class_weight=class_weights,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    X_train_normal = engineer.X_train[engineer.y_train == 0].to_numpy()
    autoencoder = AutoencoderModel(input_dim=engineer.X_train.shape[1])
    autoencoder.build()
    autoencoder.train(X_train_normal, epochs=EPOCHS, batch_size=BATCH_SIZE)

    X_test = engineer.X_test.to_numpy()
    y_test = engineer.y_test.to_numpy()

    predictions = {
        "DL 1: MLP Supervisado": mlp.predict(X_test).ravel(),
        "DL 2: Autoencoder": autoencoder.reconstruction_error(X_test),
    }

    evaluator = ModelEvaluator(y_test)
    summary = evaluator.evaluate(predictions)

    results_path = REPORTS_DIR / "model_results.csv"
    summary.to_csv(results_path, index=False)
    evaluator.plot_precision_recall(
        predictions,
        output_path=REPORTS_DIR / "figures" / "pr_curves.png",
    )

    print("=" * 80)
    print("TABLA COMPARATIVA DE MODELOS DE DEEP LEARNING")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()