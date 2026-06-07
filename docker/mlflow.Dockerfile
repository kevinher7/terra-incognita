# MLflow tracking server with S3 artifact support.
#
# The official image ships WITHOUT boto3 (mlflow/mlflow#8342), so the one and only delta
# we add is boto3 — that is everything MLflow needs for the direct client->S3 artifact path
# against floci locally and real AWS S3 in prod (contracts/mlflow-topology.md).
#
# Pinned to the SAME version the training client resolves to in uv.lock (mlflow==3.13.0),
# so server and client never drift across a 3.x API boundary.
FROM ghcr.io/mlflow/mlflow:v3.13.0

# --no-cache-dir keeps the layer small; boto3 is the sole runtime dependency we inject.
RUN pip install --no-cache-dir boto3
