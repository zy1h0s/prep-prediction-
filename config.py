"""
Configuration file for Kitchen Prep Time Prediction System.
"""
from pathlib import Path


class Config:
    """Central configuration for the prep time prediction system."""

    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"
    NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

    # Data source
    DATA_URL = "http://bit.ly/css-ml-m1-data"
    DATA_FILE = DATA_DIR / "orders.json"

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # Train/validation/test split
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

    # Data validation thresholds
    MIN_PREP_TIME = 60  # 1 minute
    MAX_PREP_TIME = 7200  # 2 hours

    # Feature engineering parameters
    TOP_N_ITEMS = 50
    TFIDF_MAX_FEATURES = 150
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95

    # Rush hours (based on typical restaurant patterns)
    RUSH_HOURS = [11, 12, 13, 17, 18, 19, 20]
    LUNCH_HOURS = [11, 12, 13, 14]
    DINNER_HOURS = [17, 18, 19, 20, 21]

    # Model training parameters
    CV_FOLDS = 5

    # XGBoost default parameters
    XGBOOST_PARAMS = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 7,
        "n_estimators": 300,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    # LightGBM default parameters
    LIGHTGBM_PARAMS = {
        "objective": "regression",
        "learning_rate": 0.05,
        "max_depth": 7,
        "n_estimators": 300,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Random Forest default parameters
    RF_PARAMS = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    # Ridge Regression parameters
    RIDGE_PARAMS = {
        "alpha": 1.0,
        "random_state": RANDOM_SEED,
    }

    # Hyperparameter tuning search space
    XGBOOST_PARAM_GRID = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 10],
        "n_estimators": [100, 300, 500],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True)
        cls.NOTEBOOK_DIR.mkdir(exist_ok=True)
