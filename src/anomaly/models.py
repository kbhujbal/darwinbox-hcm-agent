from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Anomaly:
    employee_id: str
    anomaly_type: str  # payroll_outlier | leave_abuse | compliance_violation
    confidence: float  # 0.0-1.0
    evidence: dict
    context: dict = field(default_factory=dict)  # fields the Compliance Agent needs
    suggested_action_prior: str = "flag-for-audit"

    def description(self) -> str:
        """Short natural-language summary, used as the episodic-memory embedding text."""
        return (
            f"{self.anomaly_type} for {self.employee_id}: "
            f"{', '.join(f'{k}={v}' for k, v in self.evidence.items())}"
        )
