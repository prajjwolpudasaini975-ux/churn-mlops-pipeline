import mlflow
import mlflow.sklearn

from src.pipeline import load_pipeline

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-card-churn-prediction")

# --- Logistic Regression (baseline) ---
with mlflow.start_run(run_name="logistic_regression_baseline"):
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("threshold", 0.35)
    mlflow.log_metric("accuracy", 0.91)
    mlflow.log_metric("precision", 0.75)
    mlflow.log_metric("recall", 0.70)
    mlflow.log_metric("f1", 0.72)
    mlflow.log_metric("roc_auc", 0.94)

# --- Random Forest ---
with mlflow.start_run(run_name="random_forest"):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("threshold", 0.30)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("precision", 0.80)
    mlflow.log_metric("recall", 0.87)
    mlflow.log_metric("f1", 0.84)
    mlflow.log_metric("roc_auc", 0.98)

# --- XGBoost ---
with mlflow.start_run(run_name="xgboost_untuned"):
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("threshold", 0.40)
    mlflow.log_metric("accuracy", 0.97)
    mlflow.log_metric("precision", 0.95)
    mlflow.log_metric("recall", 0.89)
    mlflow.log_metric("f1", 0.92)
    mlflow.log_metric("roc_auc", 0.99)

# --- KNN ---
with mlflow.start_run(run_name="knn"):
    mlflow.log_param("model_type", "KNN")
    mlflow.log_param("threshold", 0.2)
    mlflow.log_metric("accuracy", 0.8959)
    mlflow.log_metric("precision", 0.6532)
    mlflow.log_metric("recall", 0.7477)
    mlflow.log_metric("f1", 0.6973)
    mlflow.log_metric("roc_auc", 0.9188)

print("KNN run logged.")    

print("All 3 runs logged.")

#--- Final XGBoost pipeline: logged as an artifact + registered ---
# Uses the already-fitted, Phase 14-verified pipeline (no retraining here).
pipeline = load_pipeline()
 
with mlflow.start_run(run_name="xgboost_final_registered"):
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("threshold", 0.40)
    mlflow.log_metric("accuracy", 0.97)
    mlflow.log_metric("precision", 0.95)
    mlflow.log_metric("recall", 0.89)
    mlflow.log_metric("f1", 0.92)
    mlflow.log_metric("roc_auc", 0.99)
 
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="churn_pipeline",
        registered_model_name="churn-predictor",
        skops_trusted_types=[
            "src.features.ChurnEncoder",
            "src.features.ChurnFeatureEngineer",
            "src.features.select_columns",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )
 
print("Final pipeline logged and registered as 'churn-predictor'.")