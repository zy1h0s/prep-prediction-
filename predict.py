#!/usr/bin/env python3
"""
Prediction script for kitchen prep time prediction.

Usage:
    python predict.py --input test_orders.json --model models/xgboost.pkl
    python predict.py --input test_orders.json --model models/xgboost.pkl --output predictions.json
"""
import argparse
import logging
from pathlib import Path
import sys

from config import Config
from src.predictor import PrepTimePredictor
from src.data_loader import load_test_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict prep time for kitchen orders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict and print to console
    python predict.py --input test_orders.json --model models/xgboost.pkl

    # Predict and save to file
    python predict.py --input test_orders.json --model models/xgboost.pkl --output predictions.json

    # Use latest model
    python predict.py --input test_orders.json --model models/best_model.pkl
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSON file with orders (without prep_time_seconds)",
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model file (.pkl)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save predictions JSON file. If not specified, prints to console.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for large datasets",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip prediction validation and clipping",
    )

    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not args.model.exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("PREP TIME PREDICTION")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Model file: {args.model}")

    try:
        # Initialize predictor
        predictor = PrepTimePredictor(args.model)

        # Load test data
        logger.info("\nLoading test data...")
        test_df = load_test_data(args.input)
        logger.info(f"Loaded {len(test_df)} orders")

        # Make predictions
        logger.info("\nMaking predictions...")
        validate = not args.no_validate

        if len(test_df) > args.batch_size:
            predictions = predictor.batch_predict(
                test_df,
                batch_size=args.batch_size,
                validate=validate,
            )
        else:
            predictions = predictor.predict(test_df, validate=validate)

        # Output predictions
        if args.output:
            # Save to file
            from src.utils import save_predictions

            save_predictions(test_df["order_id"].tolist(), predictions, args.output)
            logger.info(f"\n✓ Predictions saved to {args.output}")

            # Print summary statistics
            logger.info("\nPrediction Summary:")
            logger.info(f"  Mean:   {predictions.mean():.2f} seconds ({predictions.mean()/60:.2f} minutes)")
            logger.info(f"  Median: {predictions.mean():.2f} seconds ({predictions.mean()/60:.2f} minutes)")
            logger.info(f"  Std:    {predictions.std():.2f} seconds")
            logger.info(f"  Min:    {predictions.min():.2f} seconds ({predictions.min()/60:.2f} minutes)")
            logger.info(f"  Max:    {predictions.max():.2f} seconds ({predictions.max()/60:.2f} minutes)")

        else:
            # Print to console
            logger.info("\nPredictions:")
            logger.info("-" * 80)
            for order_id, pred in zip(test_df["order_id"], predictions):
                logger.info(f"{order_id}: {pred:.2f} seconds ({pred/60:.2f} minutes)")
            logger.info("-" * 80)

            logger.info("\nPrediction Summary:")
            logger.info(f"  Total orders: {len(predictions)}")
            logger.info(f"  Mean:   {predictions.mean():.2f} seconds ({predictions.mean()/60:.2f} minutes)")
            logger.info(f"  Median: {predictions.mean():.2f} seconds ({predictions.mean()/60:.2f} minutes)")
            logger.info(f"  Std:    {predictions.std():.2f} seconds")
            logger.info(f"  Min:    {predictions.min():.2f} seconds ({predictions.min()/60:.2f} minutes)")
            logger.info(f"  Max:    {predictions.max():.2f} seconds ({predictions.max()/60:.2f} minutes)")

        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
