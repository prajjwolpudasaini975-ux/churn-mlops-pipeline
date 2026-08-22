"""
main.py

FastAPI service for the Credit Card Churn Prediction model.

Loads the verified, Phase 14-locked pipeline once at startup and exposes
a /predict endpoint that accepts customer data as JSON and returns:
    - churn_probability  : model's raw probability (0.0 to 1.0)
    - prediction         : binary label based on the committed threshold
    - threshold_used     : the threshold that drove the binary decision

Run from the project root:
    uvicorn main:app --reload

The --reload flag auto-restarts the server when code changes (development
only — remove it in production).
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from fastapi.responses import FileResponse

from src.pipeline import load_pipeline, load_metadata

# ============================================================
# Load model and metadata ONCE at startup (not per request)
# ============================================================
pipeline = load_pipeline()
metadata = load_metadata()
threshold = metadata["decision_threshold"]

# ============================================================
# Pydantic schemas: define EXACTLY what the API accepts/returns
# ============================================================

class CustomerData(BaseModel):
    """
    Input schema — one customer's raw features, exactly as they appear
    in the original dataset BEFORE any feature engineering or encoding.

    Field names must match the column names the pipeline expects.
    Pydantic validates types automatically — if someone sends a string
    where a float is expected, FastAPI returns a clear 422 error before
    our code even runs.
    """
    Customer_Age: int
    Gender: str
    Dependent_count: int
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: int
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: int
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float


class PredictionResponse(BaseModel):
    """
    Output schema — what the API sends back.
    """
    churn_probability: float
    prediction: str
    threshold_used: float


# ============================================================
# App and endpoints
# ============================================================
app = FastAPI(
    title="Credit Card Churn Prediction API",
    description="Predicts whether a credit card customer is likely to churn.",
    version="1.0.0",
)


@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {"status": "alive", "model": "churn-predictor", "version": "1.0.0"}


app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/app")                               # new endpoint
def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """
    Accepts one customer's data, runs it through the full pipeline
    (feature engineering → encoding → column selection → classifier),
    and returns the churn prediction.
    """
    # Convert Pydantic model to a single-row DataFrame
    # (the pipeline expects a DataFrame, not a dict)
    input_df = pd.DataFrame([customer.model_dump()])

    # Get probability of churn (class 1)
    churn_proba = pipeline.predict_proba(input_df)[0, 1]

    # Apply the committed threshold
    label = "Attrited Customer" if churn_proba >= threshold else "Existing Customer"

    return PredictionResponse(
        churn_probability=round(float(churn_proba), 4),
        prediction=label,
        threshold_used=threshold,
    )


