"""ExperimentConfig: a versioned YAML is the unit of an experiment, with ad-hoc overrides.

Guards the config-driven principle: hyperparameters come from a committed file (diffable,
reproducible), typos fail loudly, and CLI overrides layer on top without mutating the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from terra_incognita.experiment import ExperimentConfig, load_experiment_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_hyperparameters_from_yaml(tmp_path):
    path = _write(tmp_path, "model_arch: yolo11n.pt\nepochs: 10\nseed: 7\n")
    cfg = load_experiment_config(path)
    assert cfg.model_arch == "yolo11n.pt"
    assert cfg.epochs == 10
    assert cfg.seed == 7
    assert cfg.imgsz == 640  # unspecified -> default


def test_cli_override_layers_on_top_of_file(tmp_path):
    path = _write(tmp_path, "epochs: 50\nseed: 42\n")
    cfg = load_experiment_config(path, epochs=3, seed=None)
    assert cfg.epochs == 3  # override wins
    assert cfg.seed == 42  # None override leaves the file value untouched


def test_typo_key_fails_loudly(tmp_path):
    # `epoch` (not `epochs`) must error rather than silently use the default.
    path = _write(tmp_path, "epoch: 10\n")
    with pytest.raises(ValidationError):
        load_experiment_config(path)


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_experiment_config(path)


def test_as_mlflow_params_is_the_full_definition():
    params = ExperimentConfig().as_mlflow_params()
    assert set(params) == {"model_arch", "epochs", "imgsz", "batch", "seed", "dataset_version"}
