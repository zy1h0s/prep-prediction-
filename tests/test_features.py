"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.feature_engineering import (
    TemporalFeatureExtractor,
    OrderComplexityExtractor,
    ItemTextExtractor,
    SubtotalFeatureExtractor,
    KitchenEncoder,
    FeatureEngineer,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "order_id": ["order1", "order2", "order3"],
            "activated_at_local": pd.to_datetime(
                ["2019-10-01 12:30:00", "2019-10-01 18:45:00", "2019-10-02 09:15:00"]
            ),
            "obfuscated_item_names": [
                '["Item A", "Item B"]',
                '["Item C", "Item D", "Item E"]',
                '["Item A"]',
            ],
            "subtotal": [25.0, 45.0, 15.0],
            "kitchen_id": ["kitchen1", "kitchen2", "kitchen1"],
            "import_source": ["web", "app", "web"],
            "prep_time_seconds": [600, 900, 450],
        }
    )


class TestTemporalFeatureExtractor:
    """Test temporal feature extraction."""

    def test_extract_hour(self, sample_df):
        """Test hour extraction."""
        extractor = TemporalFeatureExtractor("activated_at_local")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        assert features.shape == (3, 8)
        # First order at 12:30
        assert features[0, 0] == 12  # hour

    def test_extract_day_of_week(self, sample_df):
        """Test day of week extraction."""
        extractor = TemporalFeatureExtractor("activated_at_local")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        # October 1, 2019 is a Tuesday (day 1)
        assert features[0, 1] == 1

    def test_weekend_indicator(self, sample_df):
        """Test weekend indicator."""
        extractor = TemporalFeatureExtractor("activated_at_local")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        # First two orders are weekdays
        assert features[0, 2] == 0
        assert features[1, 2] == 0

    def test_rush_hour_indicator(self, sample_df):
        """Test rush hour indicator."""
        extractor = TemporalFeatureExtractor("activated_at_local")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        # 12:30 is lunch rush
        assert features[0, 3] == 1
        # 18:45 is dinner rush
        assert features[1, 3] == 1
        # 9:15 is not rush hour
        assert features[2, 3] == 0


class TestOrderComplexityExtractor:
    """Test order complexity feature extraction."""

    def test_num_items(self, sample_df):
        """Test number of items extraction."""
        extractor = OrderComplexityExtractor("obfuscated_item_names")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        assert features[0, 0] == 2  # Two items
        assert features[1, 0] == 3  # Three items
        assert features[2, 0] == 1  # One item

    def test_unique_item_count(self, sample_df):
        """Test unique item count."""
        extractor = OrderComplexityExtractor("obfuscated_item_names")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        assert features[0, 3] == 2  # Two unique items
        assert features[1, 3] == 3  # Three unique items
        assert features[2, 3] == 1  # One unique item

    def test_item_diversity(self, sample_df):
        """Test item diversity score."""
        extractor = OrderComplexityExtractor("obfuscated_item_names")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        assert features[0, 4] == 1.0  # 2/2 = 1.0
        assert features[1, 4] == 1.0  # 3/3 = 1.0
        assert features[2, 4] == 1.0  # 1/1 = 1.0


class TestItemTextExtractor:
    """Test item text feature extraction."""

    def test_tfidf_extraction(self, sample_df):
        """Test TF-IDF extraction."""
        extractor = ItemTextExtractor("obfuscated_item_names", max_features=10)
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        # Should return a matrix
        assert features.shape[0] == 3
        assert features.shape[1] <= 10

    def test_empty_items(self):
        """Test handling of empty item lists."""
        df = pd.DataFrame(
            {"obfuscated_item_names": ['[]', '["Item A"]', '[]']}
        )

        extractor = ItemTextExtractor("obfuscated_item_names", max_features=10)
        extractor.fit(df)
        features = extractor.transform(df)

        # Should not crash
        assert features.shape[0] == 3


class TestSubtotalFeatureExtractor:
    """Test subtotal feature extraction."""

    def test_subtotal_features(self, sample_df):
        """Test subtotal feature extraction."""
        extractor = SubtotalFeatureExtractor("subtotal")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        # Should have 6 features
        assert features.shape == (3, 6)

        # Original subtotal
        assert features[0, 0] == 25.0

        # Log-transformed subtotal
        assert features[0, 1] == np.log1p(25.0)

    def test_subtotal_bins(self, sample_df):
        """Test subtotal binning."""
        extractor = SubtotalFeatureExtractor("subtotal")
        extractor.fit(sample_df)
        features = extractor.transform(sample_df)

        # Check that bins sum to 1 for each sample
        for i in range(3):
            bin_sum = features[i, 3] + features[i, 4] + features[i, 5]
            assert bin_sum == 1


class TestKitchenEncoder:
    """Test kitchen ID encoding."""

    def test_target_encoding(self, sample_df):
        """Test target encoding of kitchen IDs."""
        encoder = KitchenEncoder("kitchen_id")
        y = sample_df["prep_time_seconds"].values

        encoder.fit(sample_df, y)
        features = encoder.transform(sample_df)

        # Kitchen1 has orders with 600 and 450 seconds (mean = 525)
        assert features[0, 0] == 525
        assert features[2, 0] == 525

        # Kitchen2 has order with 900 seconds
        assert features[1, 0] == 900

    def test_unknown_kitchen(self, sample_df):
        """Test handling of unknown kitchen IDs."""
        encoder = KitchenEncoder("kitchen_id")
        y = sample_df["prep_time_seconds"].values

        encoder.fit(sample_df, y)

        # Create test data with unknown kitchen
        test_df = pd.DataFrame({"kitchen_id": ["unknown_kitchen"]})
        features = encoder.transform(test_df)

        # Should use global mean
        assert features[0, 0] == y.mean()


class TestFeatureEngineer:
    """Test complete feature engineering pipeline."""

    def test_fit_transform(self, sample_df):
        """Test fit and transform."""
        engineer = FeatureEngineer()
        y = sample_df["prep_time_seconds"].values

        X = engineer.fit_transform(sample_df, y)

        # Should return a matrix
        assert X.shape[0] == 3
        assert X.shape[1] > 0

        # Should be fitted
        assert engineer.fitted is True

    def test_transform_consistency(self, sample_df):
        """Test that transform produces consistent results."""
        engineer = FeatureEngineer()
        y = sample_df["prep_time_seconds"].values

        engineer.fit(sample_df, y)
        X1 = engineer.transform(sample_df)
        X2 = engineer.transform(sample_df)

        # Should be identical
        np.testing.assert_array_equal(X1, X2)

    def test_feature_names(self, sample_df):
        """Test feature name retrieval."""
        engineer = FeatureEngineer()
        y = sample_df["prep_time_seconds"].values

        engineer.fit(sample_df, y)
        feature_names = engineer.get_feature_names()

        # Should have feature names
        assert len(feature_names) > 0
        assert isinstance(feature_names, list)

    def test_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame(
            {
                "activated_at_local": pd.to_datetime(
                    ["2019-10-01 12:30:00", "2019-10-01 18:45:00"]
                ),
                "obfuscated_item_names": ['["Item A"]', '["Item B"]'],
                "subtotal": [25.0, 45.0],
                "kitchen_id": ["kitchen1", "kitchen2"],
                "import_source": ["web", "app"],
                "prep_time_seconds": [600, 900],
            }
        )

        engineer = FeatureEngineer()
        y = df["prep_time_seconds"].values

        # Should not crash with minimal data
        X = engineer.fit_transform(df, y)
        assert X.shape[0] == 2
