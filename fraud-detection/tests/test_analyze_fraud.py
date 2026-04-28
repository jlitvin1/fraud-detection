import pandas as pd
import pytest

from analyze_fraud import summarize_results


@pytest.fixture
def scored():
    return pd.DataFrame([
        {"transaction_id": 1, "amount_usd": 100.0, "risk_label": "high"},
        {"transaction_id": 2, "amount_usd": 200.0, "risk_label": "high"},
        {"transaction_id": 3, "amount_usd": 50.0,  "risk_label": "low"},
        {"transaction_id": 4, "amount_usd": 75.0,  "risk_label": "low"},
        {"transaction_id": 5, "amount_usd": 500.0, "risk_label": "medium"},
    ])


@pytest.fixture
def chargebacks():
    # Only transactions 1 and 2 are confirmed fraud
    return pd.DataFrame([{"transaction_id": 1}, {"transaction_id": 2}])


def test_sort_order_is_low_medium_high(scored, chargebacks):
    summary = summarize_results(scored, chargebacks)
    assert list(summary["risk_label"]) == ["low", "medium", "high"]


def test_transaction_counts(scored, chargebacks):
    summary = summarize_results(scored, chargebacks)
    counts = summary.set_index("risk_label")["transactions"]
    assert counts["low"] == 2
    assert counts["medium"] == 1
    assert counts["high"] == 2


def test_chargeback_rate_confirmed_fraud_group(scored, chargebacks):
    summary = summarize_results(scored, chargebacks)
    high = summary.loc[summary["risk_label"] == "high", "chargeback_rate"].iloc[0]
    assert high == 1.0


def test_chargeback_rate_clean_group(scored, chargebacks):
    summary = summarize_results(scored, chargebacks)
    low = summary.loc[summary["risk_label"] == "low", "chargeback_rate"].iloc[0]
    assert low == 0.0


def test_chargeback_rate_partial_group():
    # One fraud out of two medium transactions → rate of 0.5
    scored_partial = pd.DataFrame([
        {"transaction_id": 10, "amount_usd": 400.0, "risk_label": "medium"},
        {"transaction_id": 11, "amount_usd": 600.0, "risk_label": "medium"},
    ])
    chargebacks_partial = pd.DataFrame([{"transaction_id": 10}])
    summary = summarize_results(scored_partial, chargebacks_partial)
    rate = summary.loc[summary["risk_label"] == "medium", "chargeback_rate"].iloc[0]
    assert rate == 0.5


def test_total_amount_by_group(scored, chargebacks):
    summary = summarize_results(scored, chargebacks)
    amounts = summary.set_index("risk_label")["total_amount_usd"]
    assert amounts["high"] == 300.0
    assert amounts["low"] == 125.0
    assert amounts["medium"] == 500.0


def test_no_chargebacks_rates_are_zero():
    scored_clean = pd.DataFrame([
        {"transaction_id": 1, "amount_usd": 50.0, "risk_label": "low"},
        {"transaction_id": 2, "amount_usd": 200.0, "risk_label": "medium"},
    ])
    empty_chargebacks = pd.DataFrame([{"transaction_id": -1}])
    summary = summarize_results(scored_clean, empty_chargebacks)
    assert (summary["chargeback_rate"] == 0.0).all()
