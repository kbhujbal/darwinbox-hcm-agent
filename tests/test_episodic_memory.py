from src.anomaly.models import Anomaly
from src.rl.episodic_memory import SimilarIncident, confidence_boost


def _anomaly(anomaly_type="payroll_outlier"):
    return Anomaly(
        employee_id="EMP1",
        anomaly_type=anomaly_type,
        confidence=0.7,
        evidence={"z_score": 4.0},
        context={},
    )


def test_no_similar_incidents_gives_zero_boost():
    assert confidence_boost(_anomaly(), similar=[]) == 0.0


def test_close_positively_resolved_incident_boosts_confidence():
    similar = [
        SimilarIncident(
            incident_id="i1",
            description="payroll_outlier for EMP2",
            anomaly_type="payroll_outlier",
            action_taken="auto-correct",
            reward=1.0,
            distance=0.05,  # very close match
        )
    ]
    boost = confidence_boost(_anomaly(), similar=similar)
    assert boost > 0.0


def test_negatively_resolved_incident_gives_no_boost():
    similar = [
        SimilarIncident(
            incident_id="i1",
            description="payroll_outlier for EMP2",
            anomaly_type="payroll_outlier",
            action_taken="auto-correct",
            reward=-1.0,
            distance=0.05,
        )
    ]
    assert confidence_boost(_anomaly(), similar=similar) == 0.0


def test_different_anomaly_type_is_ignored():
    similar = [
        SimilarIncident(
            incident_id="i1",
            description="leave_abuse for EMP2",
            anomaly_type="leave_abuse",
            action_taken="escalate-to-manager",
            reward=1.0,
            distance=0.02,
        )
    ]
    assert confidence_boost(_anomaly("payroll_outlier"), similar=similar) == 0.0


def test_farther_match_boosts_less_than_closer_match():
    close = [
        SimilarIncident("i1", "d", "payroll_outlier", "auto-correct", reward=1.0, distance=0.05)
    ]
    far = [
        SimilarIncident("i2", "d", "payroll_outlier", "auto-correct", reward=1.0, distance=0.45)
    ]
    assert confidence_boost(_anomaly(), similar=close) > confidence_boost(_anomaly(), similar=far)
