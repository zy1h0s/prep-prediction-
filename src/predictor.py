"""
Prediction module for prep time prediction.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import pandas as pd
import numpy as np
import joblib

from config import Config
from src.data_loader import load_test_data
from src.utils import validate_predictions, save_predictions

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class PrepTimePredictor:
    """Handles prediction for new orders."""

    def __init__(self, model_path: Path):
        """
        Initialize PrepTimePredictor.

        Args:
            model_path: Path to the saved model file.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If model file is corrupted.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading model from {model_path}")

        try:
            model_artifact = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

        # Extract model and feature engineer
        self.model = model_artifact.get("model")
        self.feature_engineer = model_artifact.get("feature_engineer")
        self.model_type = model_artifact.get("model_type", "unknown")

        if self.model is None or self.feature_engineer is None:
            raise ValueError("Model artifact is missing required components")

        logger.info(f"Model loaded successfully (type: {self.model_type})")

    def predict(
        self,
        orders: Union[Path, pd.DataFrame, List[Dict]],
        validate: bool = True,
    ) -> np.ndarray:
        """
        Predict prep time for orders.

        Args:
            orders: Can be:
                - Path to JSON file with orders
                - DataFrame with orders
                - List of order dictionaries
            validate: If True, validate and clip predictions to reasonable range.

        Returns:
            Array of predicted prep times in seconds.

        Raises:
            ValueError: If input format is invalid or required fields are missing.
        """
        # Load and prepare data
        if isinstance(orders, Path):
            logger.info(f"Loading orders from {orders}")
            df = load_test_data(orders)
        elif isinstance(orders, pd.DataFrame):
            df = orders.copy()
        elif isinstance(orders, list):
            df = pd.DataFrame(orders)
        else:
            raise ValueError(
                "orders must be a Path, DataFrame, or list of dictionaries"
            )

        # Validate required fields
        required_fields = [
            "order_id",
            "activated_at",
            "activated_at_local",
            "cooking_or_pick_completed_at",
            "import_source",
            "kitchen_id",
            "obfuscated_item_names",
            "subtotal",
        ]

        missing_fields = set(required_fields) - set(df.columns)
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        logger.info(f"Predicting for {len(df)} orders...")

        # Ensure timestamps are parsed
        timestamp_columns = [
            "activated_at",
            "activated_at_local",
            "cooking_or_pick_completed_at",
        ]
        for col in timestamp_columns:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Transform features
        try:
            X = self.feature_engineer.transform(df)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise ValueError(f"Feature engineering failed: {e}")

        # Make predictions
        try:
            predictions = self.model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")

        # Validate and clip predictions if requested
        if validate:
            predictions = validate_predictions(predictions)

        logger.info(
            f"Predictions complete. Mean: {predictions.mean():.2f}s, "
            f"Std: {predictions.std():.2f}s, "
            f"Min: {predictions.min():.2f}s, "
            f"Max: {predictions.max():.2f}s"
        )

        return predictions

    def predict_single(self, order: Dict[str, Any], validate: bool = True) -> float:
        """
        Predict prep time for a single order.

        Args:
            order: Dictionary with order data.
            validate: If True, validate and clip prediction to reasonable range.

        Returns:
            Predicted prep time in seconds.
        """
        # Convert to DataFrame and predict
        df = pd.DataFrame([order])
        predictions = self.predict(df, validate=validate)

        return float(predictions[0])

    def predict_to_file(
        self,
        input_path: Path,
        output_path: Path,
        validate: bool = True,
    ) -> None:
        """
        Load orders from file, predict, and save to output file.

        Args:
            input_path: Path to input JSON file with orders.
            output_path: Path to save predictions JSON file.
            validate: If True, validate and clip predictions.
        """
        # Load data
        df = load_test_data(input_path)

        # Make predictions
        predictions = self.predict(df, validate=validate)

        # Save predictions
        save_predictions(df["order_id"].tolist(), predictions, output_path)

        logger.info(f"Predictions saved to {output_path}")

    def batch_predict(
        self,
        orders: pd.DataFrame,
        batch_size: int = 1000,
        validate: bool = True,
    ) -> np.ndarray:
        """
        Predict in batches for large datasets.

        Args:
            orders: DataFrame with orders.
            batch_size: Number of orders to process at once.
            validate: If True, validate and clip predictions.

        Returns:
            Array of predicted prep times.
        """
        logger.info(f"Batch predicting for {len(orders)} orders (batch_size={batch_size})")

        predictions = []

        for i in range(0, len(orders), batch_size):
            batch = orders.iloc[i : i + batch_size]
            batch_predictions = self.predict(batch, validate=validate)
            predictions.extend(batch_predictions)

            if (i + batch_size) % 5000 == 0:
                logger.info(f"Processed {min(i + batch_size, len(orders))} / {len(orders)} orders")

        return np.array(predictions)

    def get_prediction_details(
        self, order: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get detailed prediction information for debugging.

        Args:
            order: Dictionary with order data.

        Returns:
            Dictionary with prediction and feature details.
        """
        # Convert to DataFrame
        df = pd.DataFrame([order])

        # Extract features
        X = self.feature_engineer.transform(df)

        # Make prediction
        prediction = self.model.predict(X)[0]

        # Get feature names and values
        feature_names = self.feature_engineer.get_feature_names()
        features_dict = dict(zip(feature_names, X[0]))

        # Get feature importance if available
        feature_importance = None
        if hasattr(self.model, "feature_importances_"):
            feature_importance = dict(
                zip(feature_names, self.model.feature_importances_)
            )

        return {
            "order_id": order.get("order_id", "unknown"),
            "predicted_prep_time_seconds": float(prediction),
            "features": features_dict,
            "feature_importance": feature_importance,
            "model_type": self.model_type,
        }


def predict_from_cli(
    input_path: Path,
    model_path: Path,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    CLI-friendly prediction function.

    Args:
        input_path: Path to input JSON file.
        model_path: Path to trained model.
        output_path: Path to save predictions. If None, returns predictions.

    Returns:
        Array of predictions.
    """
    # Initialize predictor
    predictor = PrepTimePredictor(model_path)

    # Load data
    df = load_test_data(input_path)

    # Make predictions
    predictions = predictor.predict(df)

    # Save or return
    if output_path:
        save_predictions(df["order_id"].tolist(), predictions, output_path)
        logger.info(f"Predictions saved to {output_path}")
    else:
        # Print predictions
        for order_id, pred in zip(df["order_id"], predictions):
            print(f"{order_id}: {pred:.2f} seconds ({pred/60:.2f} minutes)")

    return predictions
