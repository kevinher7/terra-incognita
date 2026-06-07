"""Local-stack smoke test: prove the MLflow + floci(S3) artifact path works end to end.

This is the slice-3 acceptance check — "log a dummy MLflow run + write an artifact through
floci S3 and read it back". It is deliberately NOT a pytest/CI test: it needs the running
docker stack (`just up`) and the heavy ML extra (`just sync-ml`), neither of which CI has.
CI proves the *observability* path instead, with the in-memory OTel exporter (tests/).

What it actually exercises — the contract's promise (mlflow-topology.md):
  - tracking goes to the MLflow server (MLFLOW_TRACKING_URI),
  - the artifact is written *directly* to S3 by the client (no --serve-artifacts proxy),
  - and reading it back returns the exact bytes — a real round-trip, not a stat() check.

All endpoints/credentials come from the environment via Settings (loaded from .env by the
`just` recipe's dotenv) — nothing localhost is hardcoded here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow

from terra_incognita.config import Settings

_EXPERIMENT = "stack-smoke"
_ARTIFACT_NAME = "smoke.txt"


def run_smoke() -> bool:
    """Log a run + one artifact, read the artifact back, and return whether bytes match."""
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(_EXPERIMENT)

    # Unique-ish payload (git_sha) so a stale download can't accidentally pass the check.
    payload = f"stack smoke ok @ git_sha={settings.git_sha}\n"

    with tempfile.TemporaryDirectory() as tmp:
        local_src = Path(tmp) / _ARTIFACT_NAME
        local_src.write_text(payload)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_param("git_sha", settings.git_sha)
            mlflow.log_metric("dummy", 1.0)
            mlflow.log_artifact(str(local_src))  # client -> S3 (floci), direct

    # Read it back through MLflow's artifact API (resolves to the same direct-S3 download).
    client = mlflow.MlflowClient()
    listed = [a.path for a in client.list_artifacts(run_id)]
    downloaded = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=_ARTIFACT_NAME)
    roundtrip = Path(downloaded).read_text()
    ok = roundtrip == payload

    print(f"tracking_uri = {settings.mlflow_tracking_uri}")
    print(f"s3_endpoint  = {settings.s3_endpoint_url}")
    print(f"run_id       = {run_id}")
    print(f"artifacts    = {listed}")
    print(f"roundtrip_ok = {ok}")
    return ok


def main() -> int:
    ok = run_smoke()
    if ok:
        print("\nSMOKE PASS: MLflow run logged and artifact round-tripped through floci S3.")
        return 0
    print("\nSMOKE FAIL: artifact read-back did not match what was written.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
