# Master Prompt: Kitchen Order Prep Time Prediction System

Build a complete production-ready machine learning system to predict order preparation time for restaurant kitchens. The system must handle data analysis, feature engineering, model training, and serve predictions via CLI.

## Project Requirements

Create a Python-based ML system with the following structure:
```
prep-time-predictor/
├── README.md
├── requirements.txt
├── setup.py
├── config.py
├── data/
│   └── orders.json (downloaded from provided URL)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── predictor.py
│   └── utils.py
├── models/
│   └── (saved model artifacts)
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── sample_test_data.json
├── predict.py (CLI entry point)
└── train.py (training script)
```

## Data Schema

Input data fields:
- `activated_at`: UTC timestamp when order placed (string)
- `activated_at_local`: Local timestamp when order placed (string)
- `cooking_or_pick_completed_at`: UTC timestamp when cooking completed (string)
- `import_source`: "web" or "app"
- `order_id`: Unique order identifier (string)
- `kitchen_id`: Kitchen identifier (string)
- `obfuscated_item_names`: List of item names (list of strings)
- `subtotal`: Order value in USD (float)
- `prep_time_seconds`: Target variable - prep time in seconds (int) - ONLY IN TRAINING DATA

Target: Predict `prep_time_seconds` for test orders that don't have this field.

## Core Functionality

### 1. Data Loading and Validation (data_loader.py)

Implement robust data loading:
- Download data from http://bit.ly/css-ml-m1-data if not present
- Parse JSON with proper error handling
- Validate required fields exist
- Handle missing values appropriately
- Convert timestamp strings to datetime objects
- Detect and handle data quality issues (duplicates, outliers, invalid timestamps)
- Split data into train/validation/test sets (70/15/15) with time-based splitting to prevent leakage
- Save processed datasets for reproducibility

Edge cases to handle:
- Malformed JSON
- Missing required fields
- Invalid timestamp formats
- Negative or zero prep times
- Orders with completion time before activation time
- Extreme outliers (prep time > 2 hours or < 1 minute)
- Empty item lists

### 2. Exploratory Data Analysis (notebooks/exploratory_analysis.ipynb)

Create comprehensive Jupyter notebook analyzing:

**Univariate Analysis:**
- Distribution of prep_time_seconds (histogram, box plot, summary statistics)
- Count of orders by kitchen_id
- Count of orders by import_source
- Distribution of subtotal
- Distribution of order counts by hour of day and day of week
- Most common items in orders

**Bivariate Analysis:**
- Prep time vs subtotal (scatter plot, correlation)
- Prep time by kitchen_id (box plots, mean comparison)
- Prep time by import_source (box plots, t-test)
- Prep time by hour of day (line plot)
- Prep time by day of week (box plots)
- Prep time by number of items in order

**Temporal Patterns:**
- Time series of average prep time over the month
- Rush hour identification (peak order times)
- Weekend vs weekday patterns
- Kitchen-specific temporal patterns

**Text Analysis:**
- Most frequent items
- Item co-occurrence patterns
- Average prep time by specific items
- Item complexity indicators

**Key Insights to Document:**
- Which features appear most predictive
- Are there clear kitchen differences
- Time-of-day effects
- Order complexity patterns
- Data quality issues found
- Outliers and anomalies

### 3. Feature Engineering (feature_engineering.py)

Create comprehensive feature extraction pipeline using scikit-learn Pipeline and ColumnTransformer:

**Temporal Features:**
- Hour of day (0-23)
- Day of week (0-6)
- Is weekend (boolean)
- Is rush hour (boolean, based on EDA findings)
- Time since midnight in seconds
- Week of month (1-5)
- Is lunch hour (11-14)
- Is dinner hour (17-21)

**Order Complexity Features:**
- Number of items in order
- Total character length of all item names
- Average item name length
- Unique item count
- Most common item in order (one-hot encoded top 50 items)
- Item diversity score (unique items / total items)

**Text Features from obfuscated_item_names:**
- TF-IDF vectorization of item names (top 100-200 features)
- Presence of specific keywords (e.g., "wings", "chicken", "vegan")
- Count of specific item types
- Average word count per item

**Order Value Features:**
- Subtotal (as-is)
- Log-transformed subtotal
- Subtotal per item
- Subtotal bins (low/medium/high based on quartiles)

**Categorical Features:**
- Kitchen ID (target encoded or one-hot if few kitchens)
- Import source (one-hot encoded)

**Historical/Aggregated Features (if time permits):**
- Average prep time for this kitchen (rolling window)
- Order count for this kitchen in past hour
- Kitchen load indicator

**Feature Engineering Pipeline Structure:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineer:
    def fit(self, df):
        # Learn transformations from training data
        pass
    
    def transform(self, df):
        # Apply transformations to any dataset
        pass
    
    def fit_transform(self, df):
        # Fit and transform in one step
        pass
```

Handle edge cases:
- New kitchen IDs at test time (use mean encoding fallback)
- New items not seen in training
- Missing timestamp fields
- Empty item lists

### 4. Model Training (model_trainer.py)

Implement multiple regression models with proper validation:

**Model Options to Implement:**
1. Gradient Boosting (XGBoost or LightGBM) - PRIMARY MODEL
2. Random Forest - BASELINE
3. Linear Regression with regularization (Ridge/Lasso) - SIMPLE BASELINE

**Training Strategy:**
- Use time-based cross-validation (not random splits)
- Implement hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Train on 70% of data, validate on 15%, hold out 15% for final test
- Save best model with timestamp
- Log all experiments with parameters and metrics

**Hyperparameters to Tune (XGBoost/LightGBM):**
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [3, 5, 7, 10]
- n_estimators: [100, 300, 500]
- min_child_weight: [1, 3, 5]
- subsample: [0.7, 0.8, 0.9, 1.0]
- colsample_bytree: [0.7, 0.8, 0.9, 1.0]

**Model Persistence:**
- Save model using joblib or pickle
- Save feature engineering pipeline with model
- Save feature names and metadata
- Version models with timestamp or git hash

**Edge Cases:**
- Very few samples for some kitchens (regularization important)
- Class imbalance in categorical features
- High-cardinality categorical features
- Correlated features

### 5. Evaluation Metrics (model_trainer.py)

Implement comprehensive metrics:

**Primary Metrics:**
- Mean Absolute Error (MAE) - in seconds
- Root Mean Squared Error (RMSE) - in seconds
- Mean Absolute Percentage Error (MAPE) - as percentage
- R² Score

**Secondary Metrics:**
- Median Absolute Error
- 95th percentile absolute error
- Metrics stratified by kitchen_id
- Metrics stratified by time of day
- Metrics stratified by order size

**Evaluation Reports:**
- Overall performance summary
- Per-kitchen performance breakdown
- Error distribution plots
- Prediction vs actual scatter plots
- Residual analysis
- Feature importance visualization

**Business Metrics:**
- Percentage of predictions within ±2 minutes
- Percentage of predictions within ±5 minutes
- Average over-prediction and under-prediction

### 6. Prediction System (predictor.py)

Implement production-ready prediction interface:

```python
class PrepTimePredictor:
    def __init__(self, model_path):
        # Load trained model and feature pipeline
        pass
    
    def predict(self, orders_json_path):
        # Load test data, engineer features, make predictions
        # Return predictions as list or save to file
        pass
    
    def predict_single(self, order_dict):
        # Predict for single order
        pass
```

Features:
- Load model from saved artifacts
- Accept JSON file path or list of order dicts
- Apply same feature engineering as training
- Return predictions in seconds
- Handle missing fields gracefully
- Validate input format
- Log warnings for unexpected inputs
- Support batch prediction for efficiency

Error handling:
- Model file not found
- Invalid input format
- Missing required fields
- Unexpected kitchen IDs
- Timestamp parsing errors

### 7. CLI Interface (predict.py and train.py)

**train.py - Training Script:**
```bash
python train.py --data data/orders.json --output models/ --cv-folds 5 --model xgboost
```

Arguments:
- `--data`: Path to training data JSON
- `--output`: Directory to save model artifacts
- `--cv-folds`: Number of cross-validation folds
- `--model`: Model type (xgboost, lightgbm, rf)
- `--tune`: Enable hyperparameter tuning
- `--config`: Path to config file with hyperparameters

Output:
- Trained model saved to models/
- Training metrics printed to console
- Validation metrics saved to JSON
- Feature importance plot saved
- Training log saved

**predict.py - Prediction Script:**
```bash
python predict.py --input test_orders.json --model models/best_model.pkl --output predictions.json
```

Arguments:
- `--input`: Path to test data JSON (without prep_time_seconds)
- `--model`: Path to trained model file
- `--output`: Path to save predictions JSON

Output format (predictions.json):
```json
[
  {
    "order_id": "order_123",
    "predicted_prep_time_seconds": 845.3
  },
  ...
]
```

### 8. Testing (tests/)

Create comprehensive tests:

**test_features.py:**
- Test feature extraction on sample orders
- Test handling of missing values
- Test handling of new categories at test time
- Test temporal feature extraction
- Test text feature extraction
- Test feature pipeline persistence

**test_model.py:**
- Test model loading and prediction
- Test prediction output format
- Test batch prediction
- Test single order prediction
- Test error handling for invalid inputs

**sample_test_data.json:**
- Create 10-20 sample orders covering edge cases
- Include various kitchen IDs, times, item counts
- Use for manual testing

### 9. Configuration (config.py)

Centralized configuration:
```python
class Config:
    DATA_URL = "http://bit.ly/css-ml-m1-data"
    RANDOM_SEED = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    
    MIN_PREP_TIME = 60
    MAX_PREP_TIME = 7200
    
    TOP_N_ITEMS = 50
    TFIDF_MAX_FEATURES = 150
    
    MODEL_DIR = "models"
    DATA_DIR = "data"
    
    CV_FOLDS = 5
    
    XGBOOST_PARAMS = {
        "learning_rate": 0.05,
        "max_depth": 7,
        "n_estimators": 300,
        ...
    }
```

### 10. Documentation (README.md)

Comprehensive README including:

**Setup Instructions:**
```markdown
## Installation

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download data (automatic on first run or manual)

## Quick Start

### Training
python train.py --data data/orders.json --model xgboost

### Prediction
python predict.py --input test_data.json --model models/best_model.pkl --output predictions.json

## Project Structure
[Directory tree with explanations]

## Modeling Approach

### Data Analysis
[Key findings from EDA]

### Feature Engineering
[Description of features created and why]

### Model Selection
[Why XGBoost was chosen, what alternatives were tried]

### Evaluation
[Metrics used, performance achieved, validation strategy]

## Design Decisions

### Architecture
[Why this structure was chosen]

### Extensibility
[How to add new features, models, or data sources]

### Production Considerations
[How this would be deployed, monitoring needed, etc.]

## Performance

- MAE: X seconds
- RMSE: Y seconds
- R²: Z
- 95% of predictions within ±N minutes

## Future Improvements
[What could be added with more time/data]
```

## Implementation Guidelines

### Code Quality
- Write production-level Python code with type hints
- Use descriptive variable names (kitchen_order_count, not koc)
- No unnecessary comments (code should be self-documenting)
- Follow PEP 8 style guide
- Use proper exception handling with specific error messages
- Implement logging for debugging (not print statements)
- Use dataclasses or named tuples for structured data
- Implement proper __repr__ for custom classes

### ML Best Practices
- Always use random seeds for reproducibility
- Separate feature engineering from model training
- Use scikit-learn pipelines for feature transformations
- Save preprocessing pipeline with model
- Implement proper train/val/test split (time-based, not random)
- Use cross-validation for hyperparameter tuning
- Track experiments (manually or with MLflow if time permits)
- Validate predictions are in reasonable range (60-7200 seconds)

### Extensibility Requirements
The architecture must support:
- Adding new features without retraining entire pipeline
- Swapping models easily (same interface)
- Adding new kitchens at prediction time
- Real-time prediction (single order inference)
- Batch prediction (thousands of orders)
- A/B testing different models
- Model versioning and rollback
- Integration with API (though not implemented, structure should allow)

### Error Handling
Implement robust error handling for:
- Missing data files
- Corrupted JSON
- Invalid timestamp formats
- Missing required fields in input
- Model file not found
- Incompatible feature dimensions
- Out-of-memory errors for large datasets
- Network errors when downloading data

### Performance Optimization
- Use vectorized operations (pandas/numpy)
- Avoid loops where possible
- Use generators for large file processing
- Implement batch prediction efficiently
- Cache expensive computations
- Use appropriate data types (int32 vs int64)

### Dependencies (requirements.txt)
Essential libraries only:
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
pytest>=7.2.0
joblib>=1.2.0
```

## Key Edge Cases and Challenges

### Data Quality Issues
- Orders with prep time > 2 hours (likely errors or special cases)
- Orders with prep time < 60 seconds (likely data entry errors)
- Missing or null values in fields
- Duplicate order IDs
- Completion time before activation time
- Orders during kitchen closed hours

### Feature Engineering Challenges
- Obfuscated item names making text analysis harder
- High cardinality in item names (hundreds of unique items)
- New items at test time not seen during training
- New kitchens at test time
- Time zone handling for local time features
- Sparse categorical features

### Model Training Challenges
- Imbalanced kitchen representation (some kitchens have few orders)
- Temporal dependencies (orders at 7pm different from 2am)
- Non-linear relationships between features and prep time
- Outliers affecting model training
- Overfitting to specific kitchens or time periods

### Prediction Challenges
- Model drift over time (new menu items, kitchen changes)
- Handling missing fields gracefully
- Ensuring predictions are in valid range
- Handling edge cases without failing completely

### Temporal Considerations
- Use time-based train/test split (not random)
- Latest 15% of data should be test set
- Validate on middle 15%
- Train on first 70%
- This prevents data leakage from future orders

## Deliverables Checklist

- [ ] Clean, runnable code following Python best practices
- [ ] Comprehensive Jupyter notebook with EDA and insights
- [ ] Feature engineering pipeline that handles edge cases
- [ ] Trained model with saved artifacts
- [ ] CLI tools for training and prediction
- [ ] Unit tests covering critical functionality
- [ ] README with setup, usage, and design decisions
- [ ] requirements.txt with all dependencies
- [ ] Example predictions on test data
- [ ] Model evaluation report with metrics
- [ ] Feature importance analysis
- [ ] Error analysis showing where model fails

## Notes
- Focus on XGBoost or LightGBM as primary model (best for tabular data)
- Implement at least one baseline (Random Forest or Linear Regression)
- Prioritize code quality and documentation over complex features
- Make architecture extensible for interviewer modifications
- Use time-based validation to prevent leakage
- Target MAE < 180 seconds (3 minutes) as good baseline
- Include business-relevant metrics (% within X minutes)
- Handle all edge cases gracefully without crashes

Start implementation with data loading and EDA to understand data characteristics, then build feature engineering, then model training, then prediction system, then polish with tests and documentation.