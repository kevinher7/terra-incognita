#!/bin/sh
# floci ready.d hook — runs INSIDE floci once it is up, with the AWS CLI pre-pointed at the
# local endpoint (the `latest-compat` image bundles it). This is the bucket bootstrap the
# slice calls for: it makes the dataset/artifact layout exist on `just up`.
#
# Idempotent on purpose: `just up` re-runs it every boot, so `mb` may hit an existing bucket
# (ignored) and the `.keep` markers are simply re-written. The bucket name is config-driven
# (TI_S3_BUCKET, passed through from .env) — nothing hardcoded, same value the training
# client uses via Settings.s3_bucket.
set -eu

BUCKET="${TI_S3_BUCKET:-terra-incognita}"

# `|| true` so a re-run against an already-created bucket is a no-op, not a failure.
aws s3 mb "s3://${BUCKET}" 2>/dev/null || true

# Empty prefix markers so the two top-level prefixes are visible immediately:
#   - mlflow-artifacts/  -> MLflow --default-artifact-root (run artifacts, models)
#   - datasets/          -> the dataset pipeline's S3 uploads (slice 4)
printf '' | aws s3 cp - "s3://${BUCKET}/mlflow-artifacts/.keep"
printf '' | aws s3 cp - "s3://${BUCKET}/datasets/.keep"

echo "floci-init: s3://${BUCKET} ready (prefixes: mlflow-artifacts/, datasets/)"
