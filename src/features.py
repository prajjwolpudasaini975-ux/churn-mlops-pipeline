"""
features.py

Custom scikit-learn transformers for the Credit Card Churn Prediction pipeline.

This module contains ONLY reusable logic: class and function definitions.
It performs NO actions on import (no .fit() calls, no data loading, no
training). It is safe to import from anywhere — a notebook, a training
script, or a FastAPI service — without triggering any side effects.

Transformers defined here:
    - ChurnFeatureEngineer : builds engineered features (is_min_credit_limit,
                              high_risk_segment) from raw customer data.
    - ChurnEncoder          : applies fixed, domain-knowledge encoding maps
                              (ordinal + one-hot style) to categorical columns.
    - column_selector       : selects and orders the final 22 model features.

All logic here was verified in Phase 14 against the original notebook
pipeline (max prediction difference: 0.0). Do not "clean up" or refactor
values flagged with a warning comment without re-verifying against that
known-good baseline.
"""

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer


# ============================================================
# Feature Engineering
# ============================================================

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates two engineered features from raw customer data:

    - is_min_credit_limit : 1 if the customer sits at the observed credit
                             limit floor ($1438.3), else 0. This value was
                             found in EDA to correlate with elevated churn.
    - high_risk_segment   : 1 if the customer has BOTH low relationship
                             count (<=2) AND low transaction count
                             (<= trans_ct_low_), else 0. This is the
                             strongest single correlate found in the
                             entire EDA (r=0.442 with churn).

    This transformer learns nothing statistical from data in fit() beyond
    setting a fixed boundary constant — see the warning comment below.
    """

    REQUIRED_COLUMNS = ["Credit_Limit", "Total_Relationship_Count", "Total_Trans_Ct"]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ChurnFeatureEngineer":
        # DO NOT replace this with a computed quantile
        # (e.g. X['Total_Trans_Ct'].quantile(0.333)).
        # That produces 53.0, not 54 — a 1-value-off boundary that silently
        # diverges from the originally trained model. This exact mismatch
        # caused a multi-hour debugging saga in Phase 14 (root cause: an
        # off-by-one in the original tercile construction back in Phase 4/5).
        # The value 54 was confirmed via 100% match-rate against the
        # known-correct training feature. Treat it as a locked constant,
        # not something to "improve."
        self.trans_ct_low_ = 54
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X)
        X = X.copy()
        X["is_min_credit_limit"] = (X["Credit_Limit"] == 1438.3).astype(int)
        low_relationship = X["Total_Relationship_Count"] <= 2
        low_trans_ct = X["Total_Trans_Ct"] <= self.trans_ct_low_
        X["high_risk_segment"] = (low_relationship & low_trans_ct).astype(int)
        return X

    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in X.columns]
        if missing:
            raise ValueError(
                f"ChurnFeatureEngineer.transform() is missing required "
                f"column(s): {missing}. Received columns: {list(X.columns)}"
            )


# ============================================================
# Encoding
# ============================================================

class ChurnEncoder(BaseEstimator, TransformerMixin):
    """
    Applies fixed, domain-knowledge encoding maps decided in Phase 5.
    Nothing is learned from data — these mappings come from our own
    understanding of the categories, not from training-set statistics.
    """

    education_map = {
        "Uneducated": 0, "High School": 1, "Unknown": 2, "College": 3,
        "Graduate": 4, "Post-Graduate": 5, "Doctorate": 6,
    }
    income_map = {
        "Less than $40K": 0, "Unknown": 1, "$40K - $60K": 2,
        "$60K - $80K": 3, "$80K - $120K": 4, "$120K +": 5,
    }

    REQUIRED_COLUMNS = [
        "Education_Level", "Income_Category", "Gender",
        "Marital_Status", "Card_Category",
    ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ChurnEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X)
        X = X.copy()

        X["Education_Level_encoded"] = self._safe_map(
            X["Education_Level"], self.education_map, "Education_Level"
        )
        X["Income_Category_encoded"] = self._safe_map(
            X["Income_Category"], self.income_map, "Income_Category"
        )
        X["Gender_M"] = (X["Gender"] == "M").astype(int)
        X["Marital_Status_Married"] = (X["Marital_Status"] == "Married").astype(int)
        X["Marital_Status_Single"] = (X["Marital_Status"] == "Single").astype(int)
        X["Marital_Status_Unknown"] = (X["Marital_Status"] == "Unknown").astype(int)
        X["Card_Category_grouped_Other"] = (X["Card_Category"] != "Blue").astype(int)
        return X

    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in X.columns]
        if missing:
            raise ValueError(
                f"ChurnEncoder.transform() is missing required column(s): "
                f"{missing}. Received columns: {list(X.columns)}"
            )

    @staticmethod
    def _safe_map(series: pd.Series, mapping: dict, col_name: str) -> pd.Series:
        """
        Maps a categorical series using a fixed dict, raising a clear error
        if a value appears that isn't in the map (rather than silently
        producing NaN, which would flow invisibly into the model).
        """
        mapped = series.map(mapping)
        if mapped.isna().any():
            bad_values = series[mapped.isna()].unique().tolist()
            raise ValueError(
                f"{col_name} contains unrecognized value(s) {bad_values} "
                f"not present in the fixed encoding map: {list(mapping.keys())}"
            )
        return mapped


# ============================================================
# Column Selection
# ============================================================

feature_order = [
    "Customer_Age", "Dependent_count", "Months_on_book", "Total_Relationship_Count",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
    "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio", "is_min_credit_limit", "high_risk_segment",
    "Education_Level_encoded", "Income_Category_encoded", "Gender_M",
    "Marital_Status_Married", "Marital_Status_Single", "Marital_Status_Unknown",
    "Card_Category_grouped_Other",
]


def select_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and orders the final 22 features the classifier expects.
    Raises a clear error if any expected column is missing, rather than
    letting a silent KeyError surface deep inside the pipeline.
    """
    missing = [col for col in feature_order if col not in X.columns]
    if missing:
        raise ValueError(
            f"select_columns() is missing expected column(s): {missing}. "
            f"This likely means an earlier pipeline step (feature "
            f"engineering or encoding) did not run as expected."
        )
    return X[feature_order]


column_selector = FunctionTransformer(select_columns)

