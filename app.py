"""ChurnGuard: an account-level churn risk scoring demo for Hugging Face Spaces."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "churnguard_pipeline.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"
UNKNOWN_VALUE = "Not provided"

FEATURE_COLUMNS = [
    "Tenure", "City_Tier", "CC_Contacted_L12m", "Payment", "Gender",
    "Service_Score", "Account_user_count", "account_segment",
    "CC_Agent_Score", "Marital_Status", "rev_per_month", "Complain_l12m",
    "rev_growth_yoy", "coupon_used_l12m", "Day_Since_CC_connect",
    "cashback_l12m", "Login_device",
]
CATEGORICAL_COLUMNS = [
    "Payment", "Gender", "account_segment", "Marital_Status", "Login_device"
]
DEFAULT_THRESHOLD = 0.275

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "ChurnGuard model artifact is missing. Run `python train_model.py --data 'Customer Churn Data.xlsx'`."
    )

MODEL = joblib.load(MODEL_PATH)
METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}


def _normalize_value(value: Any) -> Any:
    """Convert blank/unknown UI values to NaN for the fitted imputers."""
    if value is None:
        return np.nan
    if isinstance(value, str) and (not value.strip() or value.strip() == UNKNOWN_VALUE):
        return np.nan
    return value


def build_account_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Validate and shape one account using the exact training feature order."""
    missing = [name for name in FEATURE_COLUMNS if name not in payload]
    if missing:
        raise ValueError("Missing input fields: " + ", ".join(missing))

    row: dict[str, Any] = {}
    for name in FEATURE_COLUMNS:
        value = _normalize_value(payload[name])
        if name in CATEGORICAL_COLUMNS:
            row[name] = np.nan if pd.isna(value) else str(value)
        else:
            if pd.isna(value):
                row[name] = np.nan
            else:
                try:
                    row[name] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be numeric or left blank.") from exc
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_account(payload: dict[str, Any], threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Return a 0.0–1.0 churn risk score and a threshold-based decision.

    The score is the model's positive-class output. It is useful for ranking
    accounts; it is not presented as a calibrated probability.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")
    frame = build_account_frame(payload)

    # LightGBM emits a feature-name advisory for the ColumnTransformer's matrix
    # output. The fitted preprocessing pipeline preserves the original schema.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        score = float(MODEL.predict_proba(frame)[0, 1])

    flagged = score >= float(threshold)
    if score >= 0.50:
        risk_band = "High model score"
        band_key = "high"
    elif score >= DEFAULT_THRESHOLD:
        risk_band = "Watchlist"
        band_key = "watch"
    else:
        risk_band = "Lower model score"
        band_key = "low"

    # These are service prompts based on observed fields, not explanations of
    # the model's internal decision and not causal claims.
    complaint = payload.get("Complain_l12m")
    tenure = payload.get("Tenure")
    try:
        has_complaint = float(complaint) >= 1
    except (TypeError, ValueError):
        has_complaint = False
    try:
        early_tenure = float(tenure) <= 1
    except (TypeError, ValueError):
        early_tenure = False

    if has_complaint:
        action = "Service recovery: assign an owner, resolve the complaint, and confirm the account is satisfied."
    elif early_tenure:
        action = "Onboarding support: check activation, setup friction, and whether the account has reached first value. Confirm the tenure unit before using a calendar schedule."
    elif flagged:
        action = "Proactive account review: contact the account, understand its current experience, and review service or payment friction."
    else:
        action = "Continue normal engagement and monitor the account at the next scheduled data refresh."

    return {
        "risk_score": round(score, 6),
        "risk_score_display": f"{score:.6f}" if score < 0.001 or score > 0.999 else f"{score:.3f}",
        "threshold": round(float(threshold), 3),
        "flagged_for_retention": bool(flagged),
        "prediction": "Churn risk flagged" if flagged else "Not flagged at selected threshold",
        "risk_band": risk_band,
        "risk_band_key": band_key,
        "suggested_next_step": action,
        "score_interpretation": "Model risk score on a 0.0–1.0 scale. It ranks risk; it is not a guarantee or a formally calibrated individual probability.",
    }


def _result_html(result: dict[str, Any]) -> str:
    score = result["risk_score"]
    percent = score * 100
    colors = {"high": "#F97066", "watch": "#F5B544", "low": "#34D399"}
    accent = colors[result["risk_band_key"]]
    badge = result["risk_band"].upper()
    decision = "RETENTION REVIEW RECOMMENDED" if result["flagged_for_retention"] else "NO ALERT AT THIS THRESHOLD"
    return f"""
    <div style="border:1px solid #263b4b;background:#0d1b2a;border-radius:18px;padding:24px;color:#f5f7fa;font-family:Inter,Arial,sans-serif">
      <div style="display:flex;justify-content:space-between;gap:16px;align-items:center;flex-wrap:wrap">
        <div>
          <div style="font-size:12px;letter-spacing:1.6px;color:#9fb3c8;font-weight:700">CHURNGUARD RESULT</div>
          <div style="font-size:30px;font-weight:800;margin-top:6px">{result['risk_score_display']} <span style="font-size:14px;color:#9fb3c8;font-weight:500">/ 1.000</span></div>
          <div style="margin-top:4px;color:#cbd5e1;font-size:14px">Estimated churn risk score</div>
        </div>
        <div style="background:{accent}20;border:1px solid {accent};color:{accent};padding:9px 13px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.7px">{badge}</div>
      </div>
      <div style="height:12px;background:#263b4b;border-radius:99px;margin:22px 0 8px;overflow:hidden">
        <div style="height:100%;width:{percent:.2f}%;background:{accent};border-radius:99px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;color:#9fb3c8;font-size:12px"><span>0.0 · Lower score</span><span>1.0 · Higher score</span></div>
      <div style="margin-top:20px;border-top:1px solid #263b4b;padding-top:16px;font-size:14px;font-weight:800;color:{accent}">{decision}</div>
      <div style="margin-top:6px;color:#cbd5e1;font-size:13px">Decision threshold: {result['threshold']:.3f}</div>
    </div>
    """


def predict_churn(
    tenure, city_tier, cc_contacted_l12m, payment, gender, service_score,
    account_user_count, account_segment, cc_agent_score, marital_status,
    rev_per_month, complain_l12m, rev_growth_yoy, coupon_used_l12m,
    day_since_cc_connect, cashback_l12m, login_device, threshold,
):
    payload = dict(zip(FEATURE_COLUMNS, [
        tenure, city_tier, cc_contacted_l12m, payment, gender, service_score,
        account_user_count, account_segment, cc_agent_score, marital_status,
        rev_per_month, complain_l12m, rev_growth_yoy, coupon_used_l12m,
        day_since_cc_connect, cashback_l12m, login_device,
    ]))
    try:
        result = predict_account(payload, threshold)
    except Exception as exc:  # Return a useful validation message in the interface.
        return f"<div style='color:#b42318;padding:12px'>Input error: {str(exc)}</div>", "", {}

    summary = (
        f"**{result['prediction']}**\n\n"
        f"**Suggested next step:** {result['suggested_next_step']}\n\n"
        f"**Interpretation:** {result['score_interpretation']}"
    )
    return _result_html(result), summary, result


def predict_json(payload_text: str, threshold: float):
    """JSON API wrapper. Accepts an object containing the 17 named features."""
    try:
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            raise ValueError("The JSON request must be an object with named account fields.")
        return predict_account(payload, threshold)
    except Exception as exc:
        return {"error": str(exc)}


EXAMPLES = [
    [9, 1, 16, "Debit Card", "Male", 3, 4, "Super", 3, "Married", 5, 0, 15, 1, 3, 165.25, "Mobile", DEFAULT_THRESHOLD],
    [0, 1, 30, "Debit Card", "Male", 2, 4, "Regular Plus", 3, "Single", 6, 1, 14, 0, 3, 134.07, "Mobile", DEFAULT_THRESHOLD],
    [18, 3, 12, "Credit Card", "Female", 4, 2, "Super Plus", 4, "Married", 7, 0, 18, 2, 8, 180.00, "Computer", DEFAULT_THRESHOLD],
]

with gr.Blocks(title="ChurnGuard | Account Churn Risk") as demo:
    gr.Markdown(
        """
        # CHURNGUARD
        ### Account churn risk intelligence
        **Score an account. Prioritize the next best retention conversation.**

        Enter the account's latest known information. ChurnGuard returns a risk score from **0.000 to 1.000** and a threshold-based alert.
        """
    )
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("#### 01 · Account profile")
                with gr.Row():
                    tenure = gr.Number(label="Tenure (raw source value)", value=9, precision=0, info="The source does not document the unit. Do not interpret as months or years.")
                    city_tier = gr.Dropdown([1, 2, 3], label="City tier", value=1, allow_custom_value=True)
                    account_user_count = gr.Number(label="Users linked to account", value=4, precision=0)
                with gr.Row():
                    gender = gr.Dropdown(["Male", "Female", UNKNOWN_VALUE], label="Primary customer gender", value="Male")
                    marital_status = gr.Dropdown(["Single", "Married", "Divorced", UNKNOWN_VALUE], label="Marital status", value="Married")
                    account_segment = gr.Dropdown(["Regular", "Regular Plus", "Super", "Super Plus", "HNI", UNKNOWN_VALUE], label="Account segment", value="Super")
            with gr.Group():
                gr.Markdown("#### 02 · Service and support")
                with gr.Row():
                    cc_contacted_l12m = gr.Number(label="Customer-care contacts · last 12 months", value=16, precision=0)
                    complain_l12m = gr.Dropdown([0, 1], label="Complaint in last 12 months?", value=0, info="0 = No · 1 = Yes")
                    day_since_cc_connect = gr.Number(label="Days since last care contact", value=3, precision=0)
                with gr.Row():
                    service_score = gr.Number(label="Service score", value=3, precision=0, info="Use the source system's original score scale.")
                    cc_agent_score = gr.Number(label="Customer-care agent score", value=3, precision=0, info="Use the source system's original score scale.")
            with gr.Group():
                gr.Markdown("#### 03 · Commercial and engagement")
                with gr.Row():
                    payment = gr.Dropdown(["Debit Card", "Credit Card", "E wallet", "Cash on Delivery", "UPI", UNKNOWN_VALUE], label="Preferred payment", value="Debit Card")
                    login_device = gr.Dropdown(["Mobile", "Computer", UNKNOWN_VALUE], label="Preferred login device", value="Mobile")
                with gr.Row():
                    rev_per_month = gr.Number(label="Average monthly revenue (raw value)", value=5, info="Currency/unit not specified in source.")
                    rev_growth_yoy = gr.Number(label="Revenue growth (raw value)", value=15, info="Use source value; no unit transformation is applied.")
                    cashback_l12m = gr.Number(label="Average cashback (raw value)", value=165.25, info="Currency/unit not specified in source.")
                with gr.Row():
                    coupon_used_l12m = gr.Number(label="Coupon uses · last 12 months", value=1, precision=0)
            gr.Markdown("#### 04 · Alert policy")
            threshold = gr.Slider(0.05, 0.80, value=DEFAULT_THRESHOLD, step=0.005, label="Retention alert threshold", info="0.275 is the provisional high-recall threshold selected using training out-of-fold predictions in Milestone 2.")
            predict_button = gr.Button("Calculate churn risk", variant="primary", size="lg")
        with gr.Column(scale=2):
            gr.Markdown("#### Prediction")
            result_card = gr.HTML("<div style='padding:28px;border:1px dashed #94a3b8;border-radius:16px;color:#64748b'>Your account risk result will appear here.</div>")
            result_summary = gr.Markdown("Enter the account details and select **Calculate churn risk**.")
            result_json = gr.JSON(label="Structured result · API output", visible=False)
            gr.Markdown(
                """
                <div style="padding:14px;border-radius:12px;background:#f1f5f9;color:#334155;font-size:13px">
                <b>Important:</b> The 0–1 output is a model risk score used to rank accounts. It is not a guarantee of churn or a formally calibrated individual probability. Validate data timing and model performance on future periods before production use.
                </div>
                """
            )
    predict_inputs = [
        tenure, city_tier, cc_contacted_l12m, payment, gender, service_score,
        account_user_count, account_segment, cc_agent_score, marital_status,
        rev_per_month, complain_l12m, rev_growth_yoy, coupon_used_l12m,
        day_since_cc_connect, cashback_l12m, login_device, threshold,
    ]
    predict_button.click(
        predict_churn,
        inputs=predict_inputs,
        outputs=[result_card, result_summary, result_json],
        api_name="predict_churn",
    )
    gr.Examples(examples=EXAMPLES, inputs=predict_inputs, label="Try sample accounts")

    with gr.Accordion("Developer API · JSON request", open=False):
        gr.Markdown("Use this endpoint for programmatic scoring. The JSON object must contain all 17 feature names shown below.")
        json_input = gr.Code(
            language="json",
            value=json.dumps({
                "Tenure": 9, "City_Tier": 1, "CC_Contacted_L12m": 16,
                "Payment": "Debit Card", "Gender": "Male", "Service_Score": 3,
                "Account_user_count": 4, "account_segment": "Super",
                "CC_Agent_Score": 3, "Marital_Status": "Married", "rev_per_month": 5,
                "Complain_l12m": 0, "rev_growth_yoy": 15, "coupon_used_l12m": 1,
                "Day_Since_CC_connect": 3, "cashback_l12m": 165.25,
                "Login_device": "Mobile"
            }, indent=2),
            label="Request JSON",
        )
        json_threshold = gr.Slider(0.05, 0.80, value=DEFAULT_THRESHOLD, step=0.005, label="Alert threshold")
        json_button = gr.Button("Run JSON prediction")
        json_output = gr.JSON(label="JSON response")
        json_button.click(predict_json, inputs=[json_input, json_threshold], outputs=json_output, api_name="predict_json")

    gr.Markdown(
        "---\n"
        "**ChurnGuard · Capstone ML project**  \n"
        "Account-level model · LightGBM · Milestone 2 selected configuration  \n"
        "For decision support only. Apply customer privacy, fairness, and commercial approval controls."
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue"))
