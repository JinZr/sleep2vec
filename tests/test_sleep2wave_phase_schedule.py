from __future__ import annotations

import pytest

from sleep2wave.training.phase_schedule import CURRENT_PHASE_TASK_MIX, DEFAULT_PHASE_TASK_MIX, build_phase_schedule


def test_phase_schedule_matches_expected_replay_ratios():
    assert DEFAULT_PHASE_TASK_MIX[1] == {"restoration": 0.5, "imputation": 0.5}
    assert DEFAULT_PHASE_TASK_MIX[2] == {"restoration": 0.25, "imputation": 0.25, "translation": 0.5}
    assert DEFAULT_PHASE_TASK_MIX[3] == {
        "restoration": 0.125,
        "imputation": 0.125,
        "translation": 0.25,
        "two_condition": 0.5,
    }
    assert DEFAULT_PHASE_TASK_MIX[4] == {
        "restoration": 0.1,
        "imputation": 0.1,
        "translation": 0.2,
        "two_condition": 0.3,
        "partial_full": 0.3,
    }
    assert DEFAULT_PHASE_TASK_MIX[5] == {
        "restoration": 0.2,
        "imputation": 0.2,
        "translation": 0.2,
        "two_condition": 0.2,
        "partial_full": 0.2,
    }


def test_phase_schedule_can_disable_replay_defaults():
    assert CURRENT_PHASE_TASK_MIX[3] == {"two_condition": 1.0}

    schedule = build_phase_schedule(3, replay_enabled=False)

    assert schedule.task_mix == {"two_condition": 1.0}


def test_phase_schedule_normalizes_custom_mix():
    schedule = build_phase_schedule(5, {"restoration": 1.0, "translation": 3.0})

    assert schedule.normalized() == {"restoration": 0.25, "translation": 0.75}


def test_phase_schedule_rejects_unknown_task_family():
    with pytest.raises(ValueError, match="Unsupported sleep2wave task family"):
        build_phase_schedule(1, {"unknown": 1.0})


def test_phase_schedule_rejects_phase_zero_for_diffusion():
    with pytest.raises(ValueError, match="between 1 and 5"):
        build_phase_schedule(0)
