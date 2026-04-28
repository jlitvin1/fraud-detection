from risk_rules import label_risk, score_transaction

# Baseline transaction: every signal at its lowest possible value → score 0
BASE_TX = {
    "device_risk_score": 5,
    "is_international": 0,
    "amount_usd": 10.0,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


def tx(**overrides):
    return {**BASE_TX, **overrides}


# ---------------------------------------------------------------------------
# label_risk — boundary values
# ---------------------------------------------------------------------------

def test_label_risk_low_boundaries():
    assert label_risk(0) == "low"
    assert label_risk(29) == "low"


def test_label_risk_medium_boundaries():
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"


def test_label_risk_high_boundaries():
    assert label_risk(60) == "high"
    assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — clean baseline
# ---------------------------------------------------------------------------

def test_clean_transaction_scores_zero():
    assert score_transaction(BASE_TX) == 0


# ---------------------------------------------------------------------------
# score_transaction — device risk signal
# ---------------------------------------------------------------------------

def test_high_device_risk_adds_25():
    assert score_transaction(tx(device_risk_score=70)) == 25
    assert score_transaction(tx(device_risk_score=85)) == 25


def test_medium_device_risk_adds_10():
    assert score_transaction(tx(device_risk_score=40)) == 10
    assert score_transaction(tx(device_risk_score=69)) == 10


def test_low_device_risk_adds_nothing():
    assert score_transaction(tx(device_risk_score=39)) == 0


# ---------------------------------------------------------------------------
# score_transaction — international signal
# ---------------------------------------------------------------------------

def test_international_adds_15():
    assert score_transaction(tx(is_international=1)) == 15


def test_domestic_adds_nothing():
    assert score_transaction(tx(is_international=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — amount signal
# ---------------------------------------------------------------------------

def test_large_amount_adds_25():
    assert score_transaction(tx(amount_usd=1000.0)) == 25
    assert score_transaction(tx(amount_usd=1500.0)) == 25


def test_medium_amount_adds_10():
    assert score_transaction(tx(amount_usd=500.0)) == 10
    assert score_transaction(tx(amount_usd=999.99)) == 10


def test_small_amount_adds_nothing():
    assert score_transaction(tx(amount_usd=499.99)) == 0


# ---------------------------------------------------------------------------
# score_transaction — velocity signal
# ---------------------------------------------------------------------------

def test_high_velocity_adds_20():
    assert score_transaction(tx(velocity_24h=6)) == 20
    assert score_transaction(tx(velocity_24h=10)) == 20


def test_medium_velocity_adds_5():
    assert score_transaction(tx(velocity_24h=3)) == 5
    assert score_transaction(tx(velocity_24h=5)) == 5


def test_low_velocity_adds_nothing():
    assert score_transaction(tx(velocity_24h=2)) == 0


# ---------------------------------------------------------------------------
# score_transaction — failed login signal
# ---------------------------------------------------------------------------

def test_high_login_failures_adds_20():
    assert score_transaction(tx(failed_logins_24h=5)) == 20
    assert score_transaction(tx(failed_logins_24h=8)) == 20


def test_medium_login_failures_adds_10():
    assert score_transaction(tx(failed_logins_24h=2)) == 10
    assert score_transaction(tx(failed_logins_24h=4)) == 10


def test_no_login_failures_adds_nothing():
    assert score_transaction(tx(failed_logins_24h=1)) == 0


# ---------------------------------------------------------------------------
# score_transaction — prior chargeback signal
# ---------------------------------------------------------------------------

def test_multiple_prior_chargebacks_adds_20():
    assert score_transaction(tx(prior_chargebacks=2)) == 20
    assert score_transaction(tx(prior_chargebacks=5)) == 20


def test_single_prior_chargeback_adds_5():
    assert score_transaction(tx(prior_chargebacks=1)) == 5


def test_no_prior_chargebacks_adds_nothing():
    assert score_transaction(tx(prior_chargebacks=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — score clamping
# ---------------------------------------------------------------------------

def test_score_clamped_at_100():
    # All signals at max: 25+15+25+20+20+20 = 125, must clamp to 100
    all_signals = tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1400.0,
        velocity_24h=8,
        failed_logins_24h=7,
        prior_chargebacks=2,
    )
    assert score_transaction(all_signals) == 100


# ---------------------------------------------------------------------------
# score_transaction — real transaction profiles from the dataset
# ---------------------------------------------------------------------------

def test_known_fraud_txn_50011_scores_high():
    # Worst actor in dataset: RU, device=85, $1400, velocity=8, 7 logins, 1 prior cb
    # Expected: 25+15+25+20+20+5 = 110 → clamped 100 → "high"
    txn = tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1400.0,
        velocity_24h=8,
        failed_logins_24h=7,
        prior_chargebacks=1,
    )
    assert score_transaction(txn) == 100
    assert label_risk(100) == "high"


def test_known_clean_txn_50001_scores_zero():
    # Clean domestic grocery purchase: device=8, $45, velocity=1, 0 logins, 0 prior cb
    txn = tx(
        device_risk_score=8,
        is_international=0,
        amount_usd=45.20,
        velocity_24h=1,
        failed_logins_24h=0,
        prior_chargebacks=0,
    )
    assert score_transaction(txn) == 0
    assert label_risk(0) == "low"
