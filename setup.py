"""
Setup script for Kitchen Prep Time Prediction System.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="prep-time-predictor",
    version="1.0.0",
    description="Machine learning system for predicting kitchen order preparation times",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/prep-time-predictor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "lightgbm>=3.3.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "joblib>=1.2.0",
        "requests>=2.28.0",
        "python-dateutil>=2.8.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-prep-model=train:main",
            "predict-prep-time=predict:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
