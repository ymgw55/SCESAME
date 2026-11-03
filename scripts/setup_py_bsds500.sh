#!/usr/bin/env bash
# Fetch py-bsds500 (https://github.com/Britefury/py-bsds500) at a pinned commit
# and apply this repository's patch (bug fixes and a parallel evaluation script).
# The upstream code is not redistributed in this repository.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/Britefury/py-bsds500.git"
UPSTREAM_COMMIT="79da036c6facf8f121f482b59f782551fe7a9123"
TARGET_DIR="${REPO_ROOT}/py-bsds500"

if [ -e "${TARGET_DIR}" ]; then
    echo "${TARGET_DIR} already exists. Remove it first to re-run setup." >&2
    exit 1
fi

git clone "${UPSTREAM_URL}" "${TARGET_DIR}"
git -C "${TARGET_DIR}" checkout --quiet "${UPSTREAM_COMMIT}"
git -C "${TARGET_DIR}" apply "${REPO_ROOT}/patches/py-bsds500.patch"
echo "py-bsds500 is ready at ${TARGET_DIR}"
