from app.annotate_offensive_context_audit import DECISIONS
from app.prepare_offensive_context_dataset_v1 import AUDIT_TO_DECISION, covers_goal


def test_audit_mapping_keeps_offensive_context_and_rejects_no_action() -> None:
    assert AUDIT_TO_DECISION["SEM_ACAO_OFENSIVA"] == "REJECT_NO_ACTION"
    assert AUDIT_TO_DECISION["OUTRO"] == "REJECT_NO_ACTION"
    assert AUDIT_TO_DECISION["ACAO_OFENSIVA_MAS_NAO_GOL"] == "KEEP_FOR_REVIEW"


def test_goal_coverage_uses_delivery_window() -> None:
    candidate = {"delivery_start_seconds": "100", "delivery_end_seconds": "160"}
    assert covers_goal(candidate, [120.0])
    assert not covers_goal(candidate, [161.0])


def test_annotation_decisions_keep_uncertainty_and_reject_clear_noise() -> None:
    assert DECISIONS["INCERTO"] == "KEEP_FOR_REVIEW"
    assert DECISIONS["ERRO_DE_BOLA"] == "REJECT_NO_ACTION"
