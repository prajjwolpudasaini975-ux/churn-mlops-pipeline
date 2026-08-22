"""
pipeline.py

Assembly and loading logic for the Credit Card Churn Prediction pipeline.

This module contains NO training calls and NO side effects on import.
Two distinct use cases are handled by two separate functions:

    - load_pipeline()  : loads the already-FITTED pipeline from disk.
                          This is the function used for making predictions
                          (e.g. inside a FastAPI service). Fast, safe, and
                          requires no training data.

    - build_pipeline()  : constructs a FRESH, UNTRAINED pipeline structure,
                          reading hyperparameters from model_metadata.json.
                          This exists only for retraining scenarios and is
                          NOT used in the normal prediction flow. Calling
                          this does not fit the model — the caller is
                          responsible for calling .fit() explicitly, and
                          only when they actually intend to retrain.

Single source of truth: hyperparameters live in model_metadata.json, not
duplicated as hardcoded literals here. If you retrain with different
settings, update the metadata file — build_pipeline() will pick it up
automatically.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import ChurnFeatureEngineer, ChurnEncoder, column_selector

DEFAULT_PIPELINE_PATH = "models/churn_pipeline.joblib"
DEFAULT_METADATA_PATH = "models/model_metadata.json"


def load_metadata(metadata_path: str = DEFAULT_METADATA_PATH) -> Dict[str, Any]:
    """
    Loads model_metadata.json into a dict.

    Raises a clear error if the file is missing, rather than letting a
    bare FileNotFoundError surface without context.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at '{metadata_path}'. "
            f"Expected to be run from the project root directory."
        )
    with open(path, "r") as f:
        return json.load(f)


def load_pipeline(pipeline_path: str = DEFAULT_PIPELINE_PATH) -> Pipeline:
    """
    Loads the already-fitted churn prediction pipeline from disk.

    This is the function to use for making predictions. It performs no
    training — it deserializes the exact fitted objects (trained trees,
    fitted transformer state) that were saved after Phase 14 verification.

    Returns
    -------
    Pipeline
        A fitted sklearn Pipeline: feature_engineering -> encoding ->
        column_selection -> classifier. Ready to call .predict_proba()
        or .predict() on raw customer data immediately.
    """
    path = Path(pipeline_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline file not found at '{pipeline_path}'. "
            f"Expected to be run from the project root directory."
        )
    return joblib.load(path)


def build_pipeline(metadata_path: str = DEFAULT_METADATA_PATH) -> Pipeline:
    """
    Assembles a FRESH, UNTRAINED pipeline structure for retraining.

    NOT used for prediction — the returned pipeline has not seen any data
    and calling .predict() on it will fail. The caller must explicitly
    call .fit(X_train, y_train) before using it.

    Hyperparameters are read from model_metadata.json rather than
    hardcoded here, so there is a single source of truth for the model's
    documented configuration.

    Returns
    -------
    Pipeline
        An unfitted sklearn Pipeline with the same structure as the
        verified Phase 14 pipeline.
    """
    metadata = load_metadata(metadata_path)

    if "hyperparameters" not in metadata:
        raise KeyError(
            "model_metadata.json has no 'hyperparameters' key. "
            "build_pipeline() cannot construct a classifier without it. "
            "Add a 'hyperparameters' block to the metadata file first."
        )

    hyperparameters = metadata["hyperparameters"]

    return Pipeline(steps=[
        ("feature_engineering", ChurnFeatureEngineer()),
        ("encoding", ChurnEncoder()),
        ("column_selection", column_selector),
        ("classifier", XGBClassifier(
            **hyperparameters,
            eval_metric="logloss",
            n_jobs=-1,
        )),
    ])

