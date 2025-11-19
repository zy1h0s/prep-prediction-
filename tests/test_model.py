"""
Unit tests for model training and prediction.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib

from src.model_trainer import ModelTrainer
from src.predictor import PrepTimePredictor
from src.utils import calculate_metrics


@pytest.fixture
def sample_train_df():
    """Create sample training DataFrame."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "order_id": [f"order_{i}" for i in range(n)],
            "activated_at_local": pd.date_range("2019-10-01", periods=n, freq="H"),
            "activated_at": pd.date_range("2019-10-01", periods=n, freq="H"),
            "cooking_or_pick_completed_at": pd.date_range("2019-10-01", periods=n, freq="H"),
            "obfuscated_item_names": [
                f'["Item {np.random.randint(1, 10)}", "Item {np.random.randint(1, 10)}"]'
                for _ in range(n)
            ],
            "subtotal": np.random.uniform(10, 100, n),
            "kitchen_id": np.random.choice(["kitchen1", "kitchen2", "kitchen3"], n),
            "import_source": np.random.choice(["web", "app"], n),
            "prep_time_seconds": np.random.uniform(300, 1200, n),
        }
    )


@pytest.fixture
def sample_val_df():
    """Create sample validation DataFrame."""
    np.random.seed(43)
    n = 30

    return pd.DataFrame(
        {
            "order_id": [f"order_val_{i}" for i in range(n)],
            "activated_at_local": pd.date_range("2019-10-05", periods=n, freq="H"),
            "activated_at": pd.date_range("2019-10-05", periods=n, freq="H"),
            "cooking_or_pick_completed_at": pd.date_range("2019-10-05", periods=n, freq="H"),
            "obfuscated_item_names": [
                f'["Item {np.random.randint(1, 10)}", "Item {np.random.randint(1, 10)}"]'
                for _ in range(n)
            ],
            "subtotal": np.random.uniform(10, 100, n),
            "kitchen_id": np.random.choice(["kitchen1", "kitchen2", "kitchen3"], n),
            "import_source": np.random.choice(["web", "app"], n),
            "prep_time_seconds": np.random.uniform(300, 1200, n),
        }
    )


class TestModelTrainer:
    """Test model training functionality."""

    def test_ridge_training(self, sample_train_df, sample_val_df):
        """Test training a Ridge model."""
        trainer = ModelTrainer(model_type="ridge", tune_hyperparameters=False)
        trainer.train(sample_train_df, sample_val_df)

        # Check that model is trained
        assert trainer.model is not None
        assert trainer.feature_engineer is not None
        assert trainer.train_metrics is not None
        assert trainer.val_metrics is not None

    def test_xgboost_training(self, sample_train_df, sample_val_df):
        """Test training an XGBoost model."""
        trainer = ModelTrainer(model_type="xgboost", tune_hyperparameters=False)
        trainer.train(sample_train_df, sample_val_df)

        # Check that model is trained
        assert trainer.model is not None
        assert trainer.feature_engineer is not None
        assert trainer.feature_importance is not None

    def test_model_save_load(self, sample_train_df, sample_val_df):
        """Test saving and loading a model."""
        trainer = ModelTrainer(model_type="ridge")
        trainer.train(sample_train_df, sample_val_df)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            trainer.save_model(model_path)

            # Check file exists
            assert model_path.exists()

            # Load model
            loaded_trainer = ModelTrainer.load_model(model_path)

            # Check that loaded model works
            assert loaded_trainer.model is not None
            assert loaded_trainer.feature_engineer is not None

            # Make predictions with both models
            X_val = trainer.feature_engineer.transform(sample_val_df)
            pred1 = trainer.model.predict(X_val)
            pred2 = loaded_trainer.model.predict(X_val)

            # Predictions should be identical
            np.testing.assert_array_almost_equal(pred1, pred2)

    def test_stratified_evaluation(self, sample_train_df, sample_val_df):
        """Test stratified evaluation."""
        trainer = ModelTrainer(model_type="ridge")
        trainer.train(sample_train_df, sample_val_df)

        # Evaluate by kitchen
        stratified_metrics = trainer.evaluate_stratified(
            sample_val_df, stratify_col="kitchen_id"
        )

        # Should return a DataFrame
        assert isinstance(stratified_metrics, pd.DataFrame)
        assert "kitchen_id" in stratified_metrics.columns
        assert "mae" in stratified_metrics.columns

    def test_feature_importance(self, sample_train_df, sample_val_df):
        """Test feature importance extraction."""
        trainer = ModelTrainer(model_type="xgboost")
        trainer.train(sample_train_df, sample_val_df)

        # Get feature importance
        importance_df = trainer.feature_importance

        # Check structure
        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) > 0


class TestPrepTimePredictor:
    """Test prediction functionality."""

    @pytest.fixture
    def trained_model_path(self, sample_train_df, sample_val_df):
        """Create and save a trained model for testing."""
        trainer = ModelTrainer(model_type="ridge")
        trainer.train(sample_train_df, sample_val_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            trainer.save_model(model_path)
            yield model_path

    def test_predictor_initialization(self, trained_model_path):
        """Test predictor initialization."""
        predictor = PrepTimePredictor(trained_model_path)

        assert predictor.model is not None
        assert predictor.feature_engineer is not None

    def test_predict_dataframe(self, trained_model_path, sample_val_df):
        """Test prediction from DataFrame."""
        predictor = PrepTimePredictor(trained_model_path)

        # Remove target column for prediction
        test_df = sample_val_df.drop(columns=["prep_time_seconds"])

        predictions = predictor.predict(test_df)

        # Check predictions
        assert len(predictions) == len(test_df)
        assert all(predictions > 0)

    def test_predict_single_order(self, trained_model_path, sample_val_df):
        """Test prediction for single order."""
        predictor = PrepTimePredictor(trained_model_path)

        # Get first order as dict
        order = sample_val_df.iloc[0].to_dict()
        order.pop("prep_time_seconds")

        prediction = predictor.predict_single(order)

        # Check prediction
        assert isinstance(prediction, float)
        assert prediction > 0

    def test_batch_predict(self, trained_model_path, sample_val_df):
        """Test batch prediction."""
        predictor = PrepTimePredictor(trained_model_path)

        test_df = sample_val_df.drop(columns=["prep_time_seconds"])

        predictions = predictor.batch_predict(test_df, batch_size=10)

        # Check predictions
        assert len(predictions) == len(test_df)
        assert all(predictions > 0)

    def test_prediction_validation(self, trained_model_path, sample_val_df):
        """Test that predictions are validated."""
        predictor = PrepTimePredictor(trained_model_path)

        test_df = sample_val_df.drop(columns=["prep_time_seconds"])

        # Predict with validation
        predictions = predictor.predict(test_df, validate=True)

        # All predictions should be within valid range
        assert all(predictions >= 60)  # MIN_PREP_TIME
        assert all(predictions <= 7200)  # MAX_PREP_TIME

    def test_missing_fields_error(self, trained_model_path):
        """Test that missing fields raise an error."""
        predictor = PrepTimePredictor(trained_model_path)

        # Create DataFrame with missing fields
        incomplete_df = pd.DataFrame({"order_id": ["order1"], "subtotal": [25.0]})

        with pytest.raises(ValueError):
            predictor.predict(incomplete_df)


class TestMetrics:
    """Test metrics calculation."""

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])

        metrics = calculate_metrics(y_true, y_pred)

        # Check that all metrics are present
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert "within_2_min_pct" in metrics

        # Check that MAE is reasonable
        assert metrics["mae"] == 10.0

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = y_true.copy()

        metrics = calculate_metrics(y_true, y_pred)

        # Perfect predictions should have zero error
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0
