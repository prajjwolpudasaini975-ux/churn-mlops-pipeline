"""
monitoring.py

Data drift monitoring for the Credit Card Churn Prediction model.

Compares the training data distribution (reference) against incoming
live/production data to detect when the model's input assumptions
are no longer valid.

Two drift metrics are used:
    - PSI (Population Stability Index) : numerical features
    - Jensen-Shannon Divergence        : categorical features

Drift thresholds (PSI, banking industry standard):
    PSI < 0.10  -> No significant drift. Model stable.
    PSI 0.10-0.25 -> Moderate drift. Investigate.
    PSI > 0.25  -> Major drift. Retraining likely needed.

Jensen-Shannon (0 to 1 scale):
    JS < 0.10   -> Stable
    JS 0.10-0.20 -> Moderate drift
    JS > 0.20   -> Major drift

Run from the project root:
    py monitoring.py

Drift metrics are logged to MLflow under the
'churn-model-monitoring' experiment for tracking over time.
"""

import json
import os
import numpy as np
import pandas as pd
import mlflow
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

# ============================================================
# Configuration
# ============================================================

REFERENCE_DATA_PATH = "models/reference_data.csv"
METADATA_PATH       = "models/model_metadata.json"

# Numerical features to monitor with PSI
NUMERICAL_FEATURES = [
    "Customer_Age",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

# Categorical features to monitor with Jensen-Shannon divergence
CATEGORICAL_FEATURES = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

# PSI thresholds
PSI_STABLE   = 0.10
PSI_MODERATE = 0.25

# Jensen-Shannon thresholds
JS_STABLE    = 0.10
JS_MODERATE  = 0.20

# ============================================================
# PSI Calculation
# ============================================================

def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Calculates the Population Stability Index between a reference
    and current distribution for a single numerical feature.

    PSI measures how much a distribution has shifted by comparing
    the proportion of values in each bin between the two datasets.
    A small epsilon is added to avoid division by zero or log(0).

    Parameters
    ----------
    reference : pd.Series  Training data values for this feature.
    current   : pd.Series  Live/production data values for this feature.
    bins      : int        Number of bins to divide the distribution into.

    Returns
    -------
    float  PSI value. Higher = more drift.
    """
    # Define bin edges from the reference distribution
    breakpoints = np.linspace(reference.min(), reference.max(), bins + 1)
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    # Calculate proportions in each bin
    ref_counts  = np.histogram(reference, bins=breakpoints)[0]
    curr_counts = np.histogram(current,   bins=breakpoints)[0]

    # Convert to proportions, add epsilon to avoid log(0)
    epsilon = 1e-6
    ref_proportions  = ref_counts  / len(reference)  + epsilon
    curr_proportions = curr_counts / len(current)    + epsilon

    # PSI formula: sum((current% - reference%) * ln(current% / reference%))
    psi = np.sum(
        (curr_proportions - ref_proportions) *
        np.log(curr_proportions / ref_proportions)
    )
    return float(psi)


# ============================================================
# Jensen-Shannon Divergence for Categorical Features
# ============================================================

def calculate_js_divergence(reference: pd.Series, current: pd.Series) -> float:
    """
    Calculates Jensen-Shannon divergence between two categorical
    distributions. Returns a value between 0 (identical) and 1
    (completely different).

    Handles unseen categories in live data (assigns zero probability
    in reference) without crashing.

    Parameters
    ----------
    reference : pd.Series  Training data values for this feature.
    current   : pd.Series  Live/production data values for this feature.

    Returns
    -------
    float  JS divergence score between 0 and 1.
    """
    # Get all unique categories across both datasets
    all_categories = list(set(reference.unique()) | set(current.unique()))

    # Build probability distributions over all categories
    epsilon = 1e-6
    ref_probs  = np.array([
        reference.value_counts(normalize=True).get(cat, 0) + epsilon
        for cat in all_categories
    ])
    curr_probs = np.array([
        current.value_counts(normalize=True).get(cat, 0) + epsilon
        for cat in all_categories
    ])

    # Normalize to sum to 1 (epsilon addition may shift this slightly)
    ref_probs  /= ref_probs.sum()
    curr_probs /= curr_probs.sum()

    return float(jensenshannon(ref_probs, curr_probs))


# ============================================================
# Simulate Production Data With Drift (Option B)
# ============================================================

def simulate_drifted_data(reference: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates production data that has drifted from the training
    distribution. In a real system, this would be replaced by
    actual logged API requests from the prediction service.

    Drift introduced:
    - Total_Trans_Ct    : shifted up by 40 (customers transacting much more)
    - Total_Trans_Amt   : scaled up by 1.6 (higher spend amounts)
    - Credit_Limit      : shifted up by 3000 (richer customer segment)
    - Months_Inactive_12_mon : shifted down by 1 (customers more active)
    - Income_Category   : over-represented '$120K +' (demographic shift)
    - Card_Category     : more 'Gold' cards (product mix change)

    All perturbations are clearly labelled here so the drift is
    understood to be simulated, not real -- standard practice for
    demonstrating monitoring systems in portfolio projects.
    """
    drifted = reference.copy()
    np.random.seed(42)

    n = len(drifted)

    # Numerical drift
    drifted["Total_Trans_Ct"]         = (drifted["Total_Trans_Ct"] + 40).clip(lower=0)
    drifted["Total_Trans_Amt"]        = (drifted["Total_Trans_Amt"] * 1.6).clip(lower=0)
    drifted["Credit_Limit"]           = (drifted["Credit_Limit"] + 3000).clip(lower=0)
    drifted["Months_Inactive_12_mon"] = (drifted["Months_Inactive_12_mon"] - 1).clip(lower=0)

    # Categorical drift: shift Income_Category toward higher earners
    high_income_mask = np.random.random(n) < 0.4
    drifted.loc[high_income_mask, "Income_Category"] = "$120K +"

    # Categorical drift: more Gold cards
    gold_mask = np.random.random(n) < 0.3
    drifted.loc[gold_mask, "Card_Category"] = "Gold"

    return drifted


# ============================================================
# Interpret Drift Level
# ============================================================

def interpret_psi(psi: float) -> str:
    if psi < PSI_STABLE:
        return "STABLE"
    elif psi < PSI_MODERATE:
        return "MODERATE DRIFT"
    else:
        return "MAJOR DRIFT"


def interpret_js(js: float) -> str:
    if js < JS_STABLE:
        return "STABLE"
    elif js < JS_MODERATE:
        return "MODERATE DRIFT"
    else:
        return "MAJOR DRIFT"


# ============================================================
# Main Monitoring Run
# ============================================================

def run_monitoring():
    """
    Full monitoring run:
        1. Load reference (training) data
        2. Simulate drifted production data
        3. Compute PSI for numerical features
        4. Compute JS divergence for categorical features
        5. Log all metrics to MLflow
        6. Print a human-readable drift report
        7. Trigger retraining alert if drift is widespread
    """
    print("=" * 60)
    print("CHURN MODEL DRIFT MONITORING")
    print("=" * 60)

    # Load reference data
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(
            f"Reference data not found at '{REFERENCE_DATA_PATH}'. "
            f"Run the notebook cell that saves X_train_raw to this path first."
        )
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    print(f"\nReference data loaded: {reference.shape[0]} rows, "
          f"{reference.shape[1]} columns")

    # Simulate production data with drift
    production = simulate_drifted_data(reference)
    print(f"Production data simulated: {production.shape[0]} rows "
          f"(with deliberate drift introduced)\n")

    # ── MLflow tracking ──────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-model-monitoring")

    with mlflow.start_run(run_name="drift_check"):

        drift_flags   = []   # collects features that have drifted
        results       = {}   # full results for final report

        # ── Numerical features: PSI ───────────────────────────
        print("-" * 60)
        print("NUMERICAL FEATURES (PSI)")
        print("-" * 60)
        print(f"{'Feature':<35} {'PSI':>8}  {'Status'}")
        print(f"{'-------':<35} {'---':>8}  {'------'}")

        for feature in NUMERICAL_FEATURES:
            if feature not in reference.columns:
                continue
            psi    = calculate_psi(reference[feature], production[feature])
            status = interpret_psi(psi)
            results[feature] = {"metric": "PSI", "value": psi, "status": status}

            # Log to MLflow
            mlflow.log_metric(f"psi_{feature}", round(psi, 4))

            print(f"{feature:<35} {psi:>8.4f}  {status}")

            if status != "STABLE":
                drift_flags.append(feature)

        # ── Categorical features: Jensen-Shannon ──────────────
        print()
        print("-" * 60)
        print("CATEGORICAL FEATURES (Jensen-Shannon Divergence)")
        print("-" * 60)
        print(f"{'Feature':<35} {'JS Div':>8}  {'Status'}")
        print(f"{'-------':<35} {'------':>8}  {'------'}")

        for feature in CATEGORICAL_FEATURES:
            if feature not in reference.columns:
                continue
            js     = calculate_js_divergence(reference[feature], production[feature])
            status = interpret_js(js)
            results[feature] = {"metric": "JS", "value": js, "status": status}

            # Log to MLflow
            mlflow.log_metric(f"js_{feature}", round(js, 4))

            print(f"{feature:<35} {js:>8.4f}  {status}")

            if status != "STABLE":
                drift_flags.append(feature)

        # ── Summary ───────────────────────────────────────────
        total_features  = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
        drifted_count   = len(drift_flags)
        drift_rate      = drifted_count / total_features

        mlflow.log_metric("drifted_feature_count", drifted_count)
        mlflow.log_metric("drift_rate",            round(drift_rate, 4))
        mlflow.log_param("total_features_monitored", total_features)
        mlflow.log_param("reference_data_rows",      len(reference))
        mlflow.log_param("production_data_rows",     len(production))

        print()
        print("=" * 60)
        print("DRIFT SUMMARY")
        print("=" * 60)
        print(f"Features monitored  : {total_features}")
        print(f"Features drifted    : {drifted_count}")
        print(f"Drift rate          : {drift_rate:.1%}")

        if drift_flags:
            print(f"\nDrifted features    : {', '.join(drift_flags)}")

        # ── Retraining alert ──────────────────────────────────
        print()
        if drift_rate >= 0.30:
            alert = "RETRAINING ALERT: Drift rate >= 30%. Model retraining recommended."
            mlflow.log_param("retraining_alert", "YES")
            mlflow.log_param("alert_reason",
                             f"{drifted_count}/{total_features} features drifted")
            print("!" * 60)
            print(alert)
            print("!" * 60)
            print("\nNext steps:")
            print("  1. Collect fresh labelled data from the production period")
            print("  2. Call build_pipeline() from src/pipeline.py")
            print("  3. Retrain on combined old + new data")
            print("  4. Evaluate against 'churn-predictor' v1 metrics in MLflow")
            print("  5. If improved, register as 'churn-predictor' v2")
            print("  6. Promote v2 to Production in MLflow model registry")
        else:
            mlflow.log_param("retraining_alert", "NO")
            print("Model is STABLE. No retraining needed at this time.")

        print()
        print(f"Drift metrics logged to MLflow experiment: "
              f"'churn-model-monitoring'")
        print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    run_monitoring()