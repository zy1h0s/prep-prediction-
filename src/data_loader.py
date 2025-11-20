"""
Data loading and validation module for kitchen prep time prediction.
"""
import json
import logging
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading, validation, and preprocessing."""

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to the data file. If None, uses Config.DATA_FILE.
        """
        self.data_path = data_path or Config.DATA_FILE
        self.config = Config

    def download_data(self, url: Optional[str] = None, output_path: Optional[Path] = None) -> None:
        """
        Download data from URL if not present locally.

        Args:
            url: URL to download from. If None, uses Config.DATA_URL.
            output_path: Path to save the downloaded file. If None, uses Config.DATA_FILE.
        """
        url = url or self.config.DATA_URL
        output_path = output_path or self.data_path

        if output_path.exists():
            logger.info(f"Data file already exists at {output_path}")
            return

        logger.info(f"Downloading data from {url} to {output_path}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded data to {output_path}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download data from {url}: {e}")
            raise

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from JSON file.

        Returns:
            DataFrame with raw order data.

        Raises:
            FileNotFoundError: If data file doesn't exist.
            ValueError: If JSON is malformed or empty.
        """
        if not self.data_path.exists():
            logger.warning(f"Data file not found at {self.data_path}. Attempting download...")
            self.download_data()

        logger.info(f"Loading data from {self.data_path}")

        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {self.data_path}: {e}")
            raise ValueError(f"Malformed JSON file: {e}")

        if not data:
            raise ValueError("Empty data file")

        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} orders from {self.data_path}")

        return df

    def validate_data(self, df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
        """
        Validate and clean the data.

        Args:
            df: Input DataFrame.
            require_target: If True, require prep_time_seconds column.

        Returns:
            Validated DataFrame.

        Raises:
            ValueError: If required fields are missing or data is invalid.
        """
        logger.info("Validating data...")

        # Check required fields
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

        if require_target:
            required_fields.append("prep_time_seconds")

        missing_fields = set(required_fields) - set(df.columns)
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Create a copy to avoid modifying original
        df = df.copy()

        # Check for missing values in critical fields
        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Found {null_count} null values in {field}")

        # Remove rows with null order_id or kitchen_id
        initial_len = len(df)
        df = df.dropna(subset=["order_id", "kitchen_id"])
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} rows with null order_id or kitchen_id")

        # Check for duplicate order IDs
        duplicate_count = df["order_id"].duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate order IDs. Keeping first occurrence.")
            df = df.drop_duplicates(subset="order_id", keep="first")

        # Parse obfuscated_item_names if it's a string
        if df["obfuscated_item_names"].dtype == object:
            df["obfuscated_item_names"] = df["obfuscated_item_names"].apply(self._parse_item_names)

        # Check for empty item lists
        empty_items = df["obfuscated_item_names"].apply(lambda x: len(x) == 0 if isinstance(x, list) else True)
        if empty_items.any():
            logger.warning(f"Found {empty_items.sum()} orders with empty item lists")

        # Validate subtotal
        if (df["subtotal"] <= 0).any():
            invalid_count = (df["subtotal"] <= 0).sum()
            logger.warning(f"Found {invalid_count} orders with non-positive subtotal")

        logger.info(f"Data validation complete. {len(df)} valid orders.")
        return df

    def _parse_item_names(self, item_string):
        """Parse item names from string or list."""
        if isinstance(item_string, list):
            return item_string
        if isinstance(item_string, str):
            try:
                return json.loads(item_string)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse item names: {item_string}")
                return []
        return []

    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse timestamp columns to datetime objects.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with parsed timestamps.
        """
        logger.info("Parsing timestamps...")
        df = df.copy()

        timestamp_columns = [
            "activated_at",
            "activated_at_local",
            "cooking_or_pick_completed_at",
        ]

        for col in timestamp_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception as e:
                    logger.error(f"Failed to parse {col}: {e}")

                # Check for null values after parsing
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Found {null_count} invalid timestamps in {col}")

        return df

    def clean_prep_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate prep_time_seconds.

        Args:
            df: Input DataFrame with prep_time_seconds column.

        Returns:
            DataFrame with cleaned prep times.
        """
        if "prep_time_seconds" not in df.columns:
            return df

        logger.info("Cleaning prep times...")
        df = df.copy()

        initial_len = len(df)

        # Remove negative or zero prep times
        df = df[df["prep_time_seconds"] > 0]
        removed = initial_len - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} orders with non-positive prep time")

        # Remove extreme outliers
        df = df[
            (df["prep_time_seconds"] >= self.config.MIN_PREP_TIME)
            & (df["prep_time_seconds"] <= self.config.MAX_PREP_TIME)
        ]
        outliers_removed = initial_len - removed - len(df)
        if outliers_removed > 0:
            logger.warning(
                f"Removed {outliers_removed} orders with prep time outside "
                f"[{self.config.MIN_PREP_TIME}, {self.config.MAX_PREP_TIME}] seconds"
            )

        # Check for orders where completion time is before activation time
        if all(col in df.columns for col in ["activated_at", "cooking_or_pick_completed_at"]):
            invalid_times = df["cooking_or_pick_completed_at"] < df["activated_at"]
            if invalid_times.any():
                logger.warning(
                    f"Found {invalid_times.sum()} orders where completion time is before activation time"
                )
                df = df[~invalid_times]

        logger.info(f"Prep time cleaning complete. {len(df)} orders remaining.")
        return df

    def split_data(
        self, df: pd.DataFrame, time_based: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame with 'activated_at' column.
            time_based: If True, use time-based split. If False, use random split.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        logger.info(
            f"Splitting data (time_based={time_based}): "
            f"train={self.config.TRAIN_SIZE}, val={self.config.VAL_SIZE}, test={self.config.TEST_SIZE}"
        )

        if time_based and "activated_at" in df.columns:
            # Sort by time
            df = df.sort_values("activated_at").reset_index(drop=True)

            # Calculate split indices
            n = len(df)
            train_end = int(n * self.config.TRAIN_SIZE)
            val_end = int(n * (self.config.TRAIN_SIZE + self.config.VAL_SIZE))

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()

            logger.info(
                f"Time-based split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )
        else:
            # Random split
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_SEED,
            )

            val_size_adjusted = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=self.config.RANDOM_SEED,
            )

            logger.info(
                f"Random split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )

        return train_df, val_df, test_df

    def load_and_process(
        self, save_splits: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and process data with full pipeline.

        Args:
            save_splits: If True, save train/val/test splits to disk.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # Load raw data
        df = self.load_raw_data()

        # Validate
        df = self.validate_data(df, require_target=True)

        # Parse timestamps
        df = self.parse_timestamps(df)

        # Clean prep times
        df = self.clean_prep_times(df)

        # Split data
        train_df, val_df, test_df = self.split_data(df, time_based=True)

        # Save splits if requested
        if save_splits:
            self.config.DATA_DIR.mkdir(exist_ok=True)
            train_df.to_csv(self.config.DATA_DIR / "train.csv", index=False)
            val_df.to_csv(self.config.DATA_DIR / "val.csv", index=False)
            test_df.to_csv(self.config.DATA_DIR / "test.csv", index=False)
            logger.info(f"Saved splits to {self.config.DATA_DIR}")

        return train_df, val_df, test_df


def load_test_data(data_path: Path) -> pd.DataFrame:
    """
    Load test data (without prep_time_seconds).

    Args:
        data_path: Path to test data JSON file.

    Returns:
        DataFrame with test orders.
    """
    loader = DataLoader(data_path)
    df = loader.load_raw_data()
    df = loader.validate_data(df, require_target=False)
    df = loader.parse_timestamps(df)
    return df
