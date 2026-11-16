import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import DEFAULT_THRESHOLD, FEATURE_COLUMNS, build_account_frame, explain_account, predict_account


def sample_payload():
    return {
        "Tenure": 9,
        "City_Tier": 1,
        "CC_Contacted_L12m": 16,
        "Payment": "Debit Card",
        "Gender": "Male",
        "Service_Score": 3,
        "Account_user_count": 4,
        "account_segment": "Super",
        "CC_Agent_Score": 3,
        "Marital_Status": "Married",
        "rev_per_month": 5,
        "Complain_l12m": 0,
        "rev_growth_yoy": 15,
        "coupon_used_l12m": 1,
        "Day_Since_CC_connect": 3,
        "cashback_l12m": 165.25,
        "Login_device": "Mobile",
    }


def test_frame_has_expected_schema():
    frame = build_account_frame(sample_payload())
    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.shape == (1, 17)


def test_score_is_between_zero_and_one():
    result = predict_account(sample_payload())
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["threshold"] == DEFAULT_THRESHOLD
    assert isinstance(result["flagged_for_retention"], bool)


def test_threshold_changes_only_the_decision():
    payload = sample_payload()
    payload.update({"Tenure": 0, "CC_Contacted_L12m": 30, "Service_Score": 2, "Complain_l12m": 1})
    low_threshold = predict_account(payload, threshold=0.275)
    high_threshold = predict_account(payload, threshold=0.999)
    assert math.isclose(low_threshold["risk_score"], high_threshold["risk_score"], rel_tol=0, abs_tol=1e-12)
    assert low_threshold["flagged_for_retention"] is True
    assert high_threshold["flagged_for_retention"] is False



def test_shap_explanation_returns_ranked_feature_table():
    explanation = explain_account(sample_payload(), top_n=5)
    assert "Why the model produced this score" in explanation
    assert "| Input feature | Account value | Contribution | Direction |" in explanation
    assert "do not establish cause" in explanation.lower()
