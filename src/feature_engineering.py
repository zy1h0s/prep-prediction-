"""
Feature engineering module for prep time prediction.
"""
import logging
from typing import List, Dict, Any, Optional
import json

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from timestamp."""

    def __init__(self, timestamp_col: str = "activated_at_local"):
        self.timestamp_col = timestamp_col
        self.feature_names_ = None

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        self.feature_names_ = [
            f"{self.timestamp_col}_hour",
            f"{self.timestamp_col}_day_of_week",
            f"{self.timestamp_col}_is_weekend",
            f"{self.timestamp_col}_is_rush_hour",
            f"{self.timestamp_col}_is_lunch_hour",
            f"{self.timestamp_col}_is_dinner_hour",
            f"{self.timestamp_col}_time_since_midnight",
            f"{self.timestamp_col}_week_of_month",
        ]
        return self

    def transform(self, X):
        """Extract temporal features."""
        X = X.copy()

        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(X[self.timestamp_col]):
            X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col])

        features = pd.DataFrame(index=X.index)

        features[f"{self.timestamp_col}_hour"] = X[self.timestamp_col].dt.hour
        features[f"{self.timestamp_col}_day_of_week"] = X[self.timestamp_col].dt.dayofweek
        features[f"{self.timestamp_col}_is_weekend"] = (X[self.timestamp_col].dt.dayofweek >= 5).astype(int)
        features[f"{self.timestamp_col}_is_rush_hour"] = X[self.timestamp_col].dt.hour.isin(Config.RUSH_HOURS).astype(int)
        features[f"{self.timestamp_col}_is_lunch_hour"] = X[self.timestamp_col].dt.hour.isin(Config.LUNCH_HOURS).astype(int)
        features[f"{self.timestamp_col}_is_dinner_hour"] = X[self.timestamp_col].dt.hour.isin(Config.DINNER_HOURS).astype(int)

        # Time since midnight in seconds
        features[f"{self.timestamp_col}_time_since_midnight"] = (
            X[self.timestamp_col].dt.hour * 3600
            + X[self.timestamp_col].dt.minute * 60
            + X[self.timestamp_col].dt.second
        )

        # Week of month (1-5)
        features[f"{self.timestamp_col}_week_of_month"] = (X[self.timestamp_col].dt.day - 1) // 7 + 1

        return features.values

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_


class OrderComplexityExtractor(BaseEstimator, TransformerMixin):
    """Extract order complexity features."""

    def __init__(self, item_col: str = "obfuscated_item_names"):
        self.item_col = item_col
        self.feature_names_ = None

    def fit(self, X, y=None):
        """Fit the transformer."""
        self.feature_names_ = [
            "num_items",
            "total_item_name_length",
            "avg_item_name_length",
            "unique_item_count",
            "item_diversity_score",
            "avg_words_per_item",
        ]
        return self

    def _parse_items(self, items):
        """Parse items from string or list."""
        if isinstance(items, str):
            try:
                return json.loads(items)
            except:
                return []
        elif isinstance(items, list):
            return items
        return []

    def transform(self, X):
        """Extract order complexity features."""
        X = X.copy()

        features = pd.DataFrame(index=X.index)

        # Parse items if needed
        items_list = X[self.item_col].apply(self._parse_items)

        # Number of items
        features["num_items"] = items_list.apply(len)

        # Total character length
        features["total_item_name_length"] = items_list.apply(
            lambda x: sum(len(str(item)) for item in x) if len(x) > 0 else 0
        )

        # Average item name length
        features["avg_item_name_length"] = features.apply(
            lambda row: row["total_item_name_length"] / row["num_items"]
            if row["num_items"] > 0
            else 0,
            axis=1,
        )

        # Unique item count
        features["unique_item_count"] = items_list.apply(lambda x: len(set(x)) if len(x) > 0 else 0)

        # Item diversity (unique / total)
        features["item_diversity_score"] = features.apply(
            lambda row: row["unique_item_count"] / row["num_items"]
            if row["num_items"] > 0
            else 0,
            axis=1,
        )

        # Average words per item
        features["avg_words_per_item"] = items_list.apply(
            lambda x: np.mean([len(str(item).split()) for item in x]) if len(x) > 0 else 0
        )

        return features.values

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_


class ItemTextExtractor(BaseEstimator, TransformerMixin):
    """Extract text features from item names using TF-IDF."""

    def __init__(self, item_col: str = "obfuscated_item_names", max_features: int = None):
        self.item_col = item_col
        self.max_features = max_features or Config.TFIDF_MAX_FEATURES
        self.vectorizer = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        """Fit the TF-IDF vectorizer."""
        # Parse and combine item names into single strings
        item_texts = self._prepare_texts(X)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=Config.TFIDF_MIN_DF,
            max_df=Config.TFIDF_MAX_DF,
            ngram_range=(1, 2),
            lowercase=True,
        )

        self.vectorizer.fit(item_texts)
        self.feature_names_ = [f"tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]

        return self

    def _parse_items(self, items):
        """Parse items from string or list."""
        if isinstance(items, str):
            try:
                return json.loads(items)
            except:
                return []
        elif isinstance(items, list):
            return items
        return []

    def _prepare_texts(self, X):
        """Prepare item texts for vectorization."""
        X = X.copy()
        items_list = X[self.item_col].apply(self._parse_items)

        # Combine all item names into a single string per order
        item_texts = items_list.apply(
            lambda x: " ".join(str(item) for item in x) if len(x) > 0 else ""
        )

        return item_texts

    def transform(self, X):
        """Transform to TF-IDF features."""
        item_texts = self._prepare_texts(X)
        tfidf_matrix = self.vectorizer.transform(item_texts)
        return tfidf_matrix.toarray()

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_


class SubtotalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from subtotal."""

    def __init__(self, subtotal_col: str = "subtotal"):
        self.subtotal_col = subtotal_col
        self.feature_names_ = None
        self.quartiles = None

    def fit(self, X, y=None):
        """Fit the transformer and calculate quartiles."""
        self.quartiles = X[self.subtotal_col].quantile([0.25, 0.5, 0.75]).values
        self.feature_names_ = [
            "subtotal",
            "subtotal_log",
            "subtotal_per_item",
            "subtotal_bin_low",
            "subtotal_bin_medium",
            "subtotal_bin_high",
        ]
        return self

    def transform(self, X):
        """Extract subtotal features."""
        X = X.copy()
        features = pd.DataFrame(index=X.index)

        # Original subtotal
        features["subtotal"] = X[self.subtotal_col]

        # Log-transformed subtotal (add 1 to avoid log(0))
        features["subtotal_log"] = np.log1p(X[self.subtotal_col])

        # Subtotal per item (requires num_items to be computed first, so we'll approximate)
        # This is a simplified version; in practice, we'd pass num_items
        features["subtotal_per_item"] = X[self.subtotal_col]  # Will be divided later if needed

        # Subtotal bins based on training quartiles
        if self.quartiles is not None:
            features["subtotal_bin_low"] = (X[self.subtotal_col] <= self.quartiles[0]).astype(int)
            features["subtotal_bin_medium"] = (
                (X[self.subtotal_col] > self.quartiles[0]) & (X[self.subtotal_col] <= self.quartiles[2])
            ).astype(int)
            features["subtotal_bin_high"] = (X[self.subtotal_col] > self.quartiles[2]).astype(int)
        else:
            features["subtotal_bin_low"] = 0
            features["subtotal_bin_medium"] = 0
            features["subtotal_bin_high"] = 0

        return features.values

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_


class KitchenEncoder(BaseEstimator, TransformerMixin):
    """Encode kitchen_id with target encoding."""

    def __init__(self, kitchen_col: str = "kitchen_id"):
        self.kitchen_col = kitchen_col
        self.kitchen_means = None
        self.global_mean = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        """Fit the encoder by calculating mean prep time per kitchen."""
        if y is None:
            # For test data, we can't do target encoding, so we'll use a default
            self.global_mean = 0
            self.kitchen_means = {}
        else:
            self.global_mean = y.mean()
            # Calculate mean target per kitchen
            kitchen_df = pd.DataFrame({self.kitchen_col: X[self.kitchen_col], "target": y})
            self.kitchen_means = kitchen_df.groupby(self.kitchen_col)["target"].mean().to_dict()

        self.feature_names_ = ["kitchen_id_encoded"]
        return self

    def transform(self, X):
        """Transform kitchen_id to encoded values."""
        X = X.copy()
        features = pd.DataFrame(index=X.index)

        # Use kitchen mean if available, otherwise use global mean
        features["kitchen_id_encoded"] = X[self.kitchen_col].map(
            lambda k: self.kitchen_means.get(k, self.global_mean)
        )

        return features.values

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_


class FeatureEngineer:
    """
    Main feature engineering pipeline that combines all feature extractors.
    """

    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self.fitted = False

    def _create_pipeline(self) -> Pipeline:
        """Create the feature engineering pipeline."""
        # We'll manually combine features since we need custom transformers
        # This is a simplified version; a more complex version would use ColumnTransformer
        return None  # We'll handle this manually in fit/transform

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the feature engineering pipeline.

        Args:
            df: Input DataFrame with raw features.
            y: Target variable (prep_time_seconds).

        Returns:
            self
        """
        logger.info("Fitting feature engineering pipeline...")

        # Initialize all extractors
        self.temporal_extractor = TemporalFeatureExtractor("activated_at_local")
        self.complexity_extractor = OrderComplexityExtractor("obfuscated_item_names")
        self.text_extractor = ItemTextExtractor("obfuscated_item_names")
        self.subtotal_extractor = SubtotalFeatureExtractor("subtotal")
        self.kitchen_encoder = KitchenEncoder("kitchen_id")

        # One-hot encoder for import_source
        self.import_source_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # Scaler for numerical features
        self.scaler = StandardScaler()

        # Fit each component
        self.temporal_extractor.fit(df)
        self.complexity_extractor.fit(df)
        self.text_extractor.fit(df)
        self.subtotal_extractor.fit(df)
        self.kitchen_encoder.fit(df, y)
        self.import_source_encoder.fit(df[["import_source"]])

        # Get all feature names
        self.feature_names = (
            list(self.temporal_extractor.get_feature_names_out())
            + list(self.complexity_extractor.get_feature_names_out())
            + list(self.text_extractor.get_feature_names_out())
            + list(self.subtotal_extractor.get_feature_names_out())
            + list(self.kitchen_encoder.get_feature_names_out())
            + [f"import_source_{cat}" for cat in self.import_source_encoder.categories_[0]]
        )

        # Fit scaler on training features
        features_train = self._extract_features_raw(df)
        self.scaler.fit(features_train)

        self.fitted = True
        logger.info(f"Feature engineering pipeline fitted. Total features: {len(self.feature_names)}")

        return self

    def _extract_features_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Extract raw features without scaling."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        # Extract features from each component
        temporal_features = self.temporal_extractor.transform(df)
        complexity_features = self.complexity_extractor.transform(df)
        text_features = self.text_extractor.transform(df)
        subtotal_features = self.subtotal_extractor.transform(df)
        kitchen_features = self.kitchen_encoder.transform(df)
        import_source_features = self.import_source_encoder.transform(df[["import_source"]])

        # Concatenate all features
        features = np.hstack(
            [
                temporal_features,
                complexity_features,
                text_features,
                subtotal_features,
                kitchen_features,
                import_source_features,
            ]
        )

        return features

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted pipeline.

        Args:
            df: Input DataFrame with raw features.

        Returns:
            Numpy array of engineered features.
        """
        features = self._extract_features_raw(df)
        features_scaled = self.scaler.transform(features)
        return features_scaled

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            df: Input DataFrame with raw features.
            y: Target variable.

        Returns:
            Numpy array of engineered features.
        """
        self.fit(df, y)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted first")
        return self.feature_names

    def get_feature_importance_df(self, model) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame.

        Args:
            model: Trained model with feature_importances_ attribute.

        Returns:
            DataFrame with features and their importance scores.
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")

        importance_df = pd.DataFrame(
            {"feature": self.get_feature_names(), "importance": model.feature_importances_}
        )

        return importance_df.sort_values("importance", ascending=False)
