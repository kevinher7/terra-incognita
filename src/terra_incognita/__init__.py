"""terra-incognita — camera-trap failure-mode CV training pipeline.

This package is the MLflow training + serving side of the project. Slice 1
ships the scaffold: a Typer CLI, a typed env-driven config object, and the typed
OpenTelemetry wide-event helper that later slices emit `training.run` events through.
"""

__version__ = "0.1.0"
