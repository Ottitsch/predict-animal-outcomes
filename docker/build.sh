#!/usr/bin/env bash
# Build the flow images.
#   docker/build.sh [tag] [name ...]
# With no names, builds every image. Otherwise builds only the named ones, so a
# workflow builds just what it runs (e.g. the training pipeline does not build
# the monitor/ab images). Valid names: host data-tests train robustness monitor ab
set -euo pipefail

TAG="${1:-dev}"
shift || true
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

dockerfile_for() {
    case "$1" in
        host)       echo "host.Dockerfile" ;;
        data-tests) echo "data_tests.Dockerfile" ;;
        train)      echo "train.Dockerfile" ;;
        robustness) echo "robustness.Dockerfile" ;;
        monitor)    echo "monitor.Dockerfile" ;;
        ab)         echo "ab.Dockerfile" ;;
        *) echo "unknown image: $1" >&2; exit 2 ;;
    esac
}

build() {
    local name="$1"
    local dockerfile
    dockerfile="$(dockerfile_for "$name")"
    echo ">>> building pao-${name}:${TAG}"
    docker build \
        -f "${REPO_ROOT}/docker/${dockerfile}" \
        -t "pao-${name}:${TAG}" \
        "${REPO_ROOT}"
}

names=("$@")
if [ ${#names[@]} -eq 0 ]; then
    names=(host data-tests train robustness monitor ab)
fi

for name in "${names[@]}"; do
    build "$name"
done

echo ">>> done"
