"""tests/unit/test_planner.py.
=============================
Unit tests for the SLMForge run planner and VRAM/walltime estimator (Issue 18).
"""

from __future__ import annotations

from slmforge.engine.planner import (
    DEFAULT_SHARED_BUDGET_GB,
    RunPlan,
    estimate_run,
    plan_event,
)


def test_estimate_run_tiny_model() -> None:
    """Test run estimation for tiny base models under budget."""
    plan = estimate_run(
        dataset_size=50,
        base="sshleifer/tiny-gpt2",
        epochs=1.0,
    )
    assert isinstance(plan, RunPlan)
    assert plan.vram_gb < 2.0
    assert plan.est_minutes > 0.0
    assert plan.over_budget is False
    assert plan.warning is None


def test_estimate_run_registered_base() -> None:
    """Test estimation for registered base model phi-3-mini."""
    plan = estimate_run(
        dataset_size=100,
        base="phi-3-mini",
        epochs=3.0,
    )
    assert plan.vram_gb > 2.0
    assert plan.vram_gb < DEFAULT_SHARED_BUDGET_GB
    assert plan.over_budget is False
    assert plan.warning is None


def test_warning_fires_over_budget() -> None:
    """Verify warning fires when planned VRAM > configured shared budget."""
    # Llama 3.1 8B in FP16 (use_qlora=False) with large batch size exceeds 16GB budget
    plan = estimate_run(
        dataset_size=1000,
        base="llama-3.1-8b-instruct",
        use_qlora=False,
        batch_size=8,
        seq_len=2048,
        shared_budget_gb=16.0,
    )
    assert plan.vram_gb > 16.0
    assert plan.over_budget is True
    assert plan.warning is not None
    assert "exceeds shared GPU budget" in plan.warning


def test_qlora_under_budget() -> None:
    """Verify QLoRA 4-bit keeps 8B model within 16GB GPU budget."""
    plan = estimate_run(
        dataset_size=50,
        base="llama-3.1-8b-instruct",
        use_qlora=True,
        batch_size=2,
        seq_len=512,
        shared_budget_gb=16.0,
    )
    assert plan.vram_gb <= 16.0
    assert plan.over_budget is False
    assert plan.warning is None


def test_plan_event_format() -> None:
    """Verify plan_event emits correct streaming event format."""
    plan = estimate_run(dataset_size=50, base="phi-3-mini")
    event = plan_event(plan)

    assert isinstance(event, dict)
    assert event["event"] == "plan"
    assert "data" in event
    data = event["data"]
    assert "vram_gb" in data
    assert "est_minutes" in data
    assert "over_budget" in data
    assert "shared_budget_gb" in data
    assert "warning" in data


def test_run_plan_to_dict() -> None:
    """Verify RunPlan.to_dict returns a valid dictionary."""
    plan = estimate_run(dataset_size=100, base="qwen-2.5-7b-instruct")
    plan_dict = plan.to_dict()
    assert isinstance(plan_dict, dict)
    assert "vram_gb" in plan_dict
    assert "est_minutes" in plan_dict
    assert "over_budget" in plan_dict
