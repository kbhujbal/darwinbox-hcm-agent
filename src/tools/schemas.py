"""OpenAI-style function-calling schemas for the mock HR tools.

These are passed to the LLM for structured argument extraction and also
used by executor.py to validate that a call is dispatchable.
"""

CHECK_LEAVE_BALANCE = {
    "name": "check_leave_balance",
    "description": "Check an employee's remaining leave balance for a given leave type.",
    "parameters": {
        "type": "object",
        "properties": {
            "employee_id": {
                "type": "string",
                "description": "Employee ID, e.g. 'E1001'.",
            },
            "leave_type": {
                "type": "string",
                "enum": ["earned_leave", "casual_leave", "sick_leave"],
                "description": "Which leave balance to check.",
            },
        },
        "required": ["employee_id", "leave_type"],
    },
}

APPLY_LEAVE = {
    "name": "apply_leave",
    "description": "Submit a leave application for an employee.",
    "parameters": {
        "type": "object",
        "properties": {
            "employee_id": {"type": "string", "description": "Employee ID, e.g. 'E1001'."},
            "leave_type": {
                "type": "string",
                "enum": ["earned_leave", "casual_leave", "sick_leave"],
                "description": "Type of leave being applied for.",
            },
            "start_date": {
                "type": "string",
                "description": "ISO date (YYYY-MM-DD) the leave starts.",
            },
            "num_days": {
                "type": "integer",
                "description": "Number of consecutive leave days requested.",
            },
            "reason": {
                "type": "string",
                "description": "Optional short reason for the leave.",
            },
        },
        "required": ["employee_id", "leave_type", "start_date", "num_days"],
    },
}

GET_PAYSLIP = {
    "name": "get_payslip",
    "description": "Fetch an employee's payslip breakdown for a given month.",
    "parameters": {
        "type": "object",
        "properties": {
            "employee_id": {"type": "string", "description": "Employee ID, e.g. 'E1001'."},
            "month": {
                "type": "string",
                "description": "Month in YYYY-MM format, e.g. '2026-06'.",
            },
        },
        "required": ["employee_id", "month"],
    },
}

CORRECT_PAYROLL_DISCREPANCY = {
    "name": "correct_payroll_discrepancy",
    "description": "Apply an automatic payroll correction for a detected discrepancy.",
    "parameters": {
        "type": "object",
        "properties": {
            "employee_id": {"type": "string", "description": "Employee ID, e.g. 'EMP0042'."},
            "month": {"type": "string", "description": "Month in YYYY-MM format."},
            "adjustment_amount": {
                "type": "number",
                "description": "Signed correction amount (positive = employee is owed more).",
            },
            "reason": {"type": "string", "description": "Short reason for the correction."},
        },
        "required": ["employee_id", "month", "adjustment_amount", "reason"],
    },
}

REMIND_COMPLIANCE_TRAINING = {
    "name": "remind_compliance_training",
    "description": "Send an automated reminder to complete overdue mandatory compliance training.",
    "parameters": {
        "type": "object",
        "properties": {
            "employee_id": {"type": "string", "description": "Employee ID, e.g. 'EMP0042'."},
            "training_name": {"type": "string", "description": "Name of the overdue training."},
        },
        "required": ["employee_id"],
    },
}

ALL_TOOL_SCHEMAS = [
    CHECK_LEAVE_BALANCE,
    APPLY_LEAVE,
    GET_PAYSLIP,
    CORRECT_PAYROLL_DISCREPANCY,
    REMIND_COMPLIANCE_TRAINING,
]

TOOL_SCHEMAS_BY_NAME = {schema["name"]: schema for schema in ALL_TOOL_SCHEMAS}
