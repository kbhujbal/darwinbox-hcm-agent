from src.anomaly.models import Anomaly
from src.rl.bandit import CONTEXT_DIM
from src.rl.features import encode_context


def test_context_vector_has_expected_dimension():
    anomaly = Anomaly(
        employee_id="EMP1",
        anomaly_type="payroll_outlier",
        confidence=0.9,
        evidence={"z_score": 4.5},
        context={"tenure_days": 1000},
    )
    vec = encode_context(anomaly)
    assert vec.shape == (CONTEXT_DIM,)


def test_anomaly_type_one_hot_is_exclusive():
    anomaly = Anomaly(
        employee_id="EMP1",
        anomaly_type="leave_abuse",
        confidence=0.8,
        evidence={"q1_total_leave_days": 20},
        context={},
    )
    vec = encode_context(anomaly)
    # indices 2,3,4 correspond to payroll_outlier, leave_abuse, compliance_violation
    assert vec[2] == 0.0
    assert vec[3] == 1.0
    assert vec[4] == 0.0


def test_higher_z_score_yields_higher_severity():
    low = Anomaly("EMP1", "payroll_outlier", 0.5, {"z_score": 3.6}, {})
    high = Anomaly("EMP2", "payroll_outlier", 0.9, {"z_score": 7.0}, {})
    assert encode_context(high)[5] > encode_context(low)[5]


def test_past_reward_and_prior_incident_flow_through():
    anomaly = Anomaly("EMP1", "compliance_violation", 0.7, {"reason": "training_overdue", "overdue_days": 30}, {})
    vec = encode_context(anomaly, past_similar_avg_reward=0.8, has_prior_incident=True)
    assert vec[7] == 0.8
    assert vec[8] == 1.0
