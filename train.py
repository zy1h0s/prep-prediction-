#!/usr/bin/env python3
"""
Training script for kitchen prep time prediction model.

Usage:
    python train.py --data data/orders.json --model xgboost --output models/
    python train.py --data data/orders.json --model xgboost --tune
    python train.py --compare  # Train and compare multiple models
"""
import argparse
import logging
from pathlib import Path
import sys

from config import Config
from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer, train_multiple_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train prep time prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train XGBoost model with default parameters
    python train.py --data data/orders.json --model xgboost

    # Train with hyperparameter tuning
    python train.py --data data/orders.json --model xgboost --tune

    # Train and compare multiple models
    python train.py --data data/orders.json --compare

    # Specify output directory
    python train.py --data data/orders.json --model lightgbm --output models/
        """,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Config.DATA_FILE,
        help="Path to training data JSON file",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "rf", "ridge"],
        help="Model type to train",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Config.MODEL_DIR,
        help="Directory to save model artifacts",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train and compare multiple models",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=Config.CV_FOLDS,
        help="Number of cross-validation folds for tuning",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Ensure directories exist
    Config.ensure_directories()
    args.output.mkdir(exist_ok=True)

    # Load and process data
    logger.info("=" * 80)
    logger.info("PREP TIME PREDICTION MODEL TRAINING")
    logger.info("=" * 80)

    try:
        loader = DataLoader(args.data)
        train_df, val_df, test_df = loader.load_and_process(save_splits=True)

        logger.info(f"\nDataset sizes:")
        logger.info(f"  Training:   {len(train_df)} orders")
        logger.info(f"  Validation: {len(val_df)} orders")
        logger.info(f"  Test:       {len(test_df)} orders")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Train model(s)
    try:
        if args.compare:
            # Train and compare multiple models
            logger.info("\nTraining and comparing multiple models...")
            models = ["ridge", "rf", "xgboost"]
            results = train_multiple_models(train_df, val_df, model_types=models)

            # Find best model
            best_model_type = min(
                results.keys(),
                key=lambda k: results[k].val_metrics["mae"],
            )
            best_trainer = results[best_model_type]

            logger.info(f"\nBest model: {best_model_type}")

            # Save best model
            if not args.no_save:
                model_path = best_trainer.save_model()
                logger.info(f"\nBest model saved to {model_path}")

                # Save plots
                best_trainer.plot_feature_importance(
                    save_path=args.output / f"{best_model_type}_feature_importance.png"
                )
                best_trainer.plot_predictions(
                    val_df,
                    save_path=args.output / f"{best_model_type}_predictions.png",
                )

        else:
            # Train single model
            logger.info(f"\nTraining {args.model} model...")
            trainer = ModelTrainer(
                model_type=args.model,
                tune_hyperparameters=args.tune,
            )

            trainer.train(train_df, val_df)

            # Evaluate stratified by kitchen
            logger.info("\nEvaluating by kitchen...")
            trainer.evaluate_stratified(val_df, stratify_col="kitchen_id")

            # Save model
            if not args.no_save:
                model_path = trainer.save_model()
                logger.info(f"\nModel saved to {model_path}")

                # Save plots
                trainer.plot_feature_importance(
                    save_path=args.output / f"{args.model}_feature_importance.png"
                )
                trainer.plot_predictions(
                    val_df,
                    save_path=args.output / f"{args.model}_predictions.png",
                )

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
