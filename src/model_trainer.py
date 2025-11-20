"""
Model training module for prep time prediction.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

from config import Config
from src.utils import calculate_metrics, print_metrics, calculate_stratified_metrics, get_model_path
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, evaluation, and persistence."""

    def __init__(self, model_type: str = "xgboost", tune_hyperparameters: bool = False):
        """
        Initialize ModelTrainer.

        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'rf', 'ridge').
            tune_hyperparameters: Whether to perform hyperparameter tuning.
        """
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.model = None
        self.feature_engineer = None
        self.train_metrics = None
        self.val_metrics = None
        self.feature_importance = None

    def _get_base_model(self) -> Any:
        """Get base model based on model_type."""
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(**Config.XGBOOST_PARAMS)
        elif self.model_type == "lightgbm":
            return lgb.LGBMRegressor(**Config.LIGHTGBM_PARAMS)
        elif self.model_type == "rf":
            return RandomForestRegressor(**Config.RF_PARAMS)
        elif self.model_type == "ridge":
            return Ridge(**Config.RIDGE_PARAMS)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_param_grid(self) -> Dict[str, list]:
        """Get hyperparameter grid for tuning."""
        if self.model_type == "xgboost":
            return Config.XGBOOST_PARAM_GRID
        elif self.model_type == "lightgbm":
            return {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7, 10],
                "n_estimators": [100, 300, 500],
                "num_leaves": [15, 31, 63],
                "min_child_samples": [10, 20, 30],
            }
        elif self.model_type == "rf":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif self.model_type == "ridge":
            return {"alpha": [0.1, 1.0, 10.0, 100.0]}
        else:
            return {}

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "prep_time_seconds",
    ) -> None:
        """
        Train the model.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            target_col: Name of target column.
        """
        logger.info(f"Training {self.model_type} model...")

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()

        # Extract features and target
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values

        # Fit feature engineer and transform data
        X_train = self.feature_engineer.fit_transform(train_df, y_train)
        X_val = self.feature_engineer.transform(val_df)

        logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Validation set: {X_val.shape[0]} samples")

        # Get base model
        base_model = self._get_base_model()

        # Hyperparameter tuning if requested
        if self.tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            param_grid = self._get_param_grid()

            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=Config.CV_FOLDS)

            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                random_state=Config.RANDOM_SEED,
                n_jobs=-1,
                verbose=1,
            )

            search.fit(X_train, y_train)
            self.model = search.best_estimator_

            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score (MAE): {-search.best_score_:.2f} seconds")
        else:
            # Train with default parameters
            self.model = base_model
            self.model.fit(X_train, y_train)

        # Evaluate on train and validation sets
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        self.train_metrics = calculate_metrics(y_train, y_train_pred)
        self.val_metrics = calculate_metrics(y_val, y_val_pred)

        # Print metrics
        print_metrics(self.train_metrics, "Training Set")
        print_metrics(self.val_metrics, "Validation Set")

        # Extract feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = self.feature_engineer.get_feature_importance_df(self.model)
            logger.info(f"\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10).to_string(index=False))

        logger.info("Training complete!")

    def evaluate_stratified(
        self,
        val_df: pd.DataFrame,
        stratify_col: str,
        target_col: str = "prep_time_seconds",
    ) -> pd.DataFrame:
        """
        Evaluate model with stratified metrics.

        Args:
            val_df: Validation DataFrame.
            stratify_col: Column to stratify by (e.g., 'kitchen_id', 'hour').
            target_col: Name of target column.

        Returns:
            DataFrame with stratified metrics.
        """
        if self.model is None or self.feature_engineer is None:
            raise ValueError("Model must be trained before evaluation")

        # Transform features
        X_val = self.feature_engineer.transform(val_df)
        y_pred = self.model.predict(X_val)

        # Create evaluation DataFrame
        eval_df = val_df.copy()
        eval_df["y_true"] = eval_df[target_col]
        eval_df["y_pred"] = y_pred

        # Calculate stratified metrics
        stratified_metrics = calculate_stratified_metrics(
            eval_df, "y_true", "y_pred", stratify_col
        )

        print(f"\nStratified Metrics by {stratify_col}:")
        print(stratified_metrics.to_string(index=False))

        return stratified_metrics

    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[Path] = None) -> None:
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to plot.
            save_path: Path to save the plot. If None, displays the plot.
        """
        if self.feature_importance is None:
            logger.warning("No feature importance available")
            return

        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)

        sns.barplot(data=top_features, x="importance", y="feature")
        plt.title(f"Top {top_n} Feature Importance - {self.model_type}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_predictions(
        self,
        val_df: pd.DataFrame,
        target_col: str = "prep_time_seconds",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot predicted vs actual values.

        Args:
            val_df: Validation DataFrame.
            target_col: Name of target column.
            save_path: Path to save the plot. If None, displays the plot.
        """
        if self.model is None or self.feature_engineer is None:
            raise ValueError("Model must be trained before plotting")

        X_val = self.feature_engineer.transform(val_df)
        y_true = val_df[target_col].values
        y_pred = self.model.predict(X_val)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
        axes[0].set_xlabel("Actual Prep Time (seconds)")
        axes[0].set_ylabel("Predicted Prep Time (seconds)")
        axes[0].set_title("Predicted vs Actual")
        axes[0].grid(True, alpha=0.3)

        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[1].set_xlabel("Predicted Prep Time (seconds)")
        axes[1].set_ylabel("Residuals (seconds)")
        axes[1].set_title("Residual Plot")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Predictions plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def save_model(self, model_path: Optional[Path] = None) -> Path:
        """
        Save trained model and feature engineer.

        Args:
            model_path: Path to save the model. If None, generates a default path.

        Returns:
            Path where model was saved.
        """
        if self.model is None or self.feature_engineer is None:
            raise ValueError("Model must be trained before saving")

        if model_path is None:
            model_path = get_model_path(self.model_type, timestamp=True)

        # Save model, feature engineer, and metadata
        model_artifact = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "model_type": self.model_type,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "feature_importance": self.feature_importance.to_dict() if self.feature_importance is not None else None,
            "timestamp": datetime.now().isoformat(),
        }

        joblib.dump(model_artifact, model_path)
        logger.info(f"Model saved to {model_path}")

        # Also save metrics to JSON
        metrics_path = model_path.with_suffix(".json")
        metrics_data = {
            "model_type": self.model_type,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

        return model_path

    @classmethod
    def load_model(cls, model_path: Path) -> "ModelTrainer":
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model.

        Returns:
            ModelTrainer instance with loaded model.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading model from {model_path}")

        model_artifact = joblib.load(model_path)

        trainer = cls(model_type=model_artifact["model_type"])
        trainer.model = model_artifact["model"]
        trainer.feature_engineer = model_artifact["feature_engineer"]
        trainer.train_metrics = model_artifact.get("train_metrics")
        trainer.val_metrics = model_artifact.get("val_metrics")

        if model_artifact.get("feature_importance") is not None:
            trainer.feature_importance = pd.DataFrame(model_artifact["feature_importance"])

        logger.info(f"Model loaded successfully")

        return trainer


def train_multiple_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_types: list = ["ridge", "rf", "xgboost"],
    target_col: str = "prep_time_seconds",
) -> Dict[str, ModelTrainer]:
    """
    Train multiple models for comparison.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        model_types: List of model types to train.
        target_col: Name of target column.

    Returns:
        Dictionary mapping model type to trained ModelTrainer.
    """
    results = {}

    for model_type in model_types:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Training {model_type} model")
        logger.info(f"{'=' * 80}")

        trainer = ModelTrainer(model_type=model_type, tune_hyperparameters=False)
        trainer.train(train_df, val_df, target_col)

        results[model_type] = trainer

    # Print comparison
    print(f"\n{'=' * 80}")
    print("Model Comparison - Validation Set")
    print(f"{'=' * 80}")

    comparison_data = []
    for model_type, trainer in results.items():
        comparison_data.append(
            {
                "Model": model_type,
                "MAE": trainer.val_metrics["mae"],
                "RMSE": trainer.val_metrics["rmse"],
                "R2": trainer.val_metrics["r2"],
                "Within 2min %": trainer.val_metrics["within_2_min_pct"],
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    return results
