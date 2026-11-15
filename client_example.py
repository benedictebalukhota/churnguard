"""Call a deployed ChurnGuard Hugging Face Space from another Python app.

Install the client with: pip install gradio_client
Set CHURNGUARD_SPACE to 'your-hf-username/your-space-name'.
"""
from __future__ import annotations

import json
import os
from gradio_client import Client

SPACE_ID = os.environ.get("CHURNGUARD_SPACE", "your-hf-username/churnguard")
HF_TOKEN = os.environ.get("HF_TOKEN")  # Only needed for a private/protected Space.

payload = {
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

client = Client(SPACE_ID, token=HF_TOKEN)
result = client.predict(json.dumps(payload), 0.275, api_name="/predict_json")
print(result)
