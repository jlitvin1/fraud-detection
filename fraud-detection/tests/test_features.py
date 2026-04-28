import pandas as pd
import pytest

from features import build_model_frame


@pytest.fixture
def sample_data():
    transactions = pd.DataFrame([
        {"transaction_id": 1, "account_id": 10, "amount_usd": 1200.0, "failed_logins_24h": 3},
        {"transaction_id": 2, "account_id": 20, "amount_usd": 400.0,  "failed_logins_24h": 2},
        {"transaction_id": 3, "account_id": 30, "amount_usd": 80.0,   "failed_logins_24h": 0},
    ])
    accounts = pd.DataFrame([
        {"account_id": 10, "prior_chargebacks": 2},
        {"account_id": 20, "prior_chargebacks": 1},
        {"account_id": 30, "prior_chargebacks": 0},
    ])
    return transactions, accounts


def test_merge_brings_in_prior_chargebacks(sample_data):
    transactions, accounts = sample_data
    df = build_model_frame(transactions, accounts)
    assert df.loc[df["account_id"] == 10, "prior_chargebacks"].iloc[0] == 2
    assert df.loc[df["account_id"] == 20, "prior_chargebacks"].iloc[0] == 1
    assert df.loc[df["account_id"] == 30, "prior_chargebacks"].iloc[0] == 0


def test_row_count_preserved(sample_data):
    transactions, accounts = sample_data
    df = build_model_frame(transactions, accounts)
    assert len(df) == len(transactions)


def test_is_large_amount_above_threshold(sample_data):
    transactions, accounts = sample_data
    df = build_model_frame(transactions, accounts)
    assert df.loc[df["transaction_id"] == 1, "is_large_amount"].iloc[0] == 1


def test_is_large_amount_below_threshold(sample_data):
    transactions, accounts = sample_data
    df = build_model_frame(transactions, accounts)
    assert df.loc[df["transaction_id"] == 2, "is_large_amount"].iloc[0] == 0
    assert df.loc[df["transaction_id"] == 3, "is_large_amount"].iloc[0] == 0


def test_login_pressure_high(sample_data):
    # failed_logins_24h=3 falls in (2, 100] → "high"
    transactions, accounts = sample_data
    df = build_model_frame(transactions, accounts)
    assert df.loc[df["transaction_id"] == 1, "login_pressure"].iloc[0] == "high"


def test_login_pressure_low(sample_data):
    # failed_logins_24h=2 falls in (0, 2] → "low"
    transactions, accounts = sample_data
    df = build_model_frame(transactions, accounts)
    assert df.loc[df["transaction_id"] == 2, "login_pressure"].iloc[0] == "low"


def test_login_pressure_none(sample_data):
    # failed_logins_24h=0 falls in (-1, 0] → "none"
    transactions, accounts = sample_data
    df = build_model_frame(transactions, accounts)
    assert df.loc[df["transaction_id"] == 3, "login_pressure"].iloc[0] == "none"
