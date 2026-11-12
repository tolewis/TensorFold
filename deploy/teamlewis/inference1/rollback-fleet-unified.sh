#!/usr/bin/env bash
# Restore inference1 Qwen prod before a fleet-unified swap test.
# Usage: ./rollback-fleet-unified.sh [YYYYMMDD]
# Default points at the 2026-11-11 p4g6 rollback bundle.
set -euo pipefail
TAG="${1:-20261111}"
exec "$HOME/b70/qwen27/rollback-p4g6.sh" "$TAG"
