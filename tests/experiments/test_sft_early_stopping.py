"""Validation convergence determines duration; the best checkpoint includes baseline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from sft_train import ValidationStopper


def test_plateau_patience_resets_after_progress():
    monitor = ValidationStopper(patience=3, min_delta=.01, min_epochs=1)
    assert monitor.observe(1, 0, 0) == (True, False)
    assert monitor.observe(.8, 8, 1) == (True, False)
    assert monitor.observe(.81, 9, 1.125) == (False, False)
    assert monitor.observe(.805, 10, 1.25) == (False, False)
    assert monitor.observe(.7, 11, 1.375) == (True, False)
    assert monitor.bad_checks == 0
    assert monitor.observe(.71, 12, 1.5) == (False, False)
    assert monitor.observe(.72, 13, 1.625) == (False, False)
    assert monitor.observe(.73, 14, 1.75) == (False, True)
    assert monitor.best_step == 11
    assert monitor.best_loss == .7


def test_small_progress_accumulates_and_true_minimum_is_saved():
    monitor = ValidationStopper(patience=3, min_delta=.01, min_epochs=0)
    monitor.observe(1, 0, 0)
    assert monitor.observe(.996, 1, .1) == (True, False)
    assert monitor.observe(.992, 2, .2) == (True, False)
    assert monitor.observe(.988, 3, .3) == (True, False)
    assert monitor.bad_checks == 0
    assert monitor.observe(.9879, 4, .4) == (True, False)
    assert monitor.observe(.9878, 5, .5) == (True, False)
    assert monitor.observe(.9877, 6, .6) == (True, True)
    assert monitor.best_step == 6


def test_warmup_does_not_count_plateau_and_baseline_can_win():
    monitor = ValidationStopper(patience=2, min_epochs=1)
    monitor.observe(1, 0, 0)
    for step in range(1, 8):
        assert monitor.observe(2, step, step / 8) == (False, False)
        assert monitor.bad_checks == 0
    assert monitor.observe(2, 8, 1) == (False, False)
    assert monitor.observe(2, 9, 1.125) == (False, True)
    assert monitor.best_step == 0 and monitor.best_loss == 1


def test_early_stopping_disabled_preserves_fixed_budget_mode():
    monitor = ValidationStopper(patience=0)
    monitor.observe(1, 0, 0)
    for step in range(1, 30):
        assert monitor.observe(2, step, step) == (False, False)


@pytest.mark.parametrize('loss', [float('nan'), float('inf'), -float('inf')])
def test_invalid_loss_fails_instead_of_reporting_convergence(loss):
    with pytest.raises(ValueError, match='non-finite validation loss'):
        ValidationStopper().observe(loss, 1, 1)


@pytest.mark.parametrize('settings', [{'patience': -1}, {'min_delta': -.1}, {'min_epochs': -1},
                                     {'min_delta': float('nan')}, {'min_epochs': float('inf')}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        ValidationStopper(**settings)
