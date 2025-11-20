"""
Utility functions for the prep time prediction system.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 95th percentile absolute error
    percentile_95 = np.percentile(np.abs(y_true - y_pred), 95)

    # Business metrics: % within N minutes
    errors = np.abs(y_true - y_pred)
    within_2_min = (errors <= 120).mean() * 100
    within_5_min = (errors <= 300).mean() * 100

    # Over/under prediction
    over_prediction = (y_pred > y_true).mean() * 100
    under_prediction = (y_pred < y_true).mean() * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": median_ae,
        "mape": mape,
        "percentile_95_ae": percentile_95,
        "within_2_min_pct": within_2_min,
        "within_5_min_pct": within_5_min,
        "over_prediction_pct": over_prediction,
        "under_prediction_pct": under_prediction,
    }


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics.
        dataset_name: Name of the dataset for display.
    """
    print(f"\n{'=' * 60}")
    print(f"{dataset_name} Performance Metrics")
    print(f"{'=' * 60}")
    print(f"MAE:                {metrics['mae']:.2f} seconds ({metrics['mae']/60:.2f} minutes)")
    print(f"RMSE:               {metrics['rmse']:.2f} seconds ({metrics['rmse']/60:.2f} minutes)")
    print(f"Median AE:          {metrics['median_ae']:.2f} seconds ({metrics['median_ae']/60:.2f} minutes)")
    print(f"MAPE:               {metrics['mape']:.2f}%")
    print(f"R² Score:           {metrics['r2']:.4f}")
    print(f"95th Percentile AE: {metrics['percentile_95_ae']:.2f} seconds")
    print(f"\nBusiness Metrics:")
    print(f"  Within ±2 min:    {metrics['within_2_min_pct']:.2f}%")
    print(f"  Within ±5 min:    {metrics['within_5_min_pct']:.2f}%")
    print(f"  Over-prediction:  {metrics['over_prediction_pct']:.2f}%")
    print(f"  Under-prediction: {metrics['under_prediction_pct']:.2f}%")
    print(f"{'=' * 60}\n")


def calculate_stratified_metrics(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, stratify_col: str
) -> pd.DataFrame:
    """
    Calculate metrics stratified by a categorical column.

    Args:
        df: DataFrame with true and predicted values.
        y_true_col: Name of true value column.
        y_pred_col: Name of predicted value column.
        stratify_col: Column to stratify by.

    Returns:
        DataFrame with metrics per category.
    """
    results = []

    for category in df[stratify_col].unique():
        mask = df[stratify_col] == category
        y_true = df.loc[mask, y_true_col].values
        y_pred = df.loc[mask, y_pred_col].values

        if len(y_true) > 0:
            metrics = calculate_metrics(y_true, y_pred)
            metrics[stratify_col] = category
            metrics["count"] = len(y_true)
            results.append(metrics)

    return pd.DataFrame(results)


def save_predictions(
    order_ids: List[str], predictions: np.ndarray, output_path: Path
) -> None:
    """
    Save predictions in the required JSON format.

    Args:
        order_ids: List of order IDs.
        predictions: Array of predicted prep times.
        output_path: Path to save the JSON file.
    """
    predictions_list = [
        {"order_id": order_id, "predicted_prep_time_seconds": float(pred)}
        for order_id, pred in zip(order_ids, predictions)
    ]

    with open(output_path, "w") as f:
        json.dump(predictions_list, f, indent=2)

    logger.info(f"Saved {len(predictions_list)} predictions to {output_path}")


def get_model_path(model_name: str, timestamp: bool = True) -> Path:
    """
    Generate model file path.

    Args:
        model_name: Name of the model (e.g., 'xgboost', 'rf').
        timestamp: If True, include timestamp in filename.

    Returns:
        Path object for the model file.
    """
    Config.MODEL_DIR.mkdir(exist_ok=True)

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{ts}.pkl"
    else:
        filename = f"{model_name}.pkl"

    return Config.MODEL_DIR / filename


def validate_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    Validate and clip predictions to reasonable range.

    Args:
        predictions: Array of predictions.

    Returns:
        Validated predictions.
    """
    # Clip to valid range
    predictions_clipped = np.clip(
        predictions, Config.MIN_PREP_TIME, Config.MAX_PREP_TIME
    )

    # Log if any predictions were clipped
    clipped_count = (predictions != predictions_clipped).sum()
    if clipped_count > 0:
        logger.warning(
            f"Clipped {clipped_count} predictions to range "
            f"[{Config.MIN_PREP_TIME}, {Config.MAX_PREP_TIME}]"
        )

    return predictions_clipped


def create_time_features_dict(timestamp: pd.Timestamp) -> Dict[str, Any]:
    """
    Extract time features from a timestamp for a single order.

    Args:
        timestamp: Pandas Timestamp object.

    Returns:
        Dictionary of time features.
    """
    return {
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "is_weekend": timestamp.dayofweek >= 5,
        "is_rush_hour": timestamp.hour in Config.RUSH_HOURS,
        "is_lunch_hour": timestamp.hour in Config.LUNCH_HOURS,
        "is_dinner_hour": timestamp.hour in Config.DINNER_HOURS,
        "time_since_midnight": timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second,
        "week_of_month": (timestamp.day - 1) // 7 + 1,
    }
