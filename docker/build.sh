#!/usr/bin/env bash
# Build all four images for the containerized flow.
# Usage: docker/build.sh [tag]   (default tag: dev)
set -euo pipefail

TAG="${1:-dev}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

build() {
    local name="$1"
    local dockerfile="$2"
    echo ">>> building pao-${name}:${TAG}"
    docker build \
        -f "${REPO_ROOT}/docker/${dockerfile}" \
        -t "pao-${name}:${TAG}" \
        "${REPO_ROOT}"
}

build host       host.Dockerfile
build data-tests data_tests.Dockerfile
build train      train.Dockerfile
build robustness robustness.Dockerfile

echo ">>> done"
