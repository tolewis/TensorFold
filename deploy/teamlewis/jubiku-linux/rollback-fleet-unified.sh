#!/usr/bin/env bash
# Roll jubiku Qwen env and binary back after a failed fleet-unified swap.
set -euo pipefail
cd "$HOME/b70/qwen27"
cp qwen27.env.bak-p1w3-20261110 qwen27.env
# Binary name must match the env QWEN_BIN line after copy.
systemctl --user restart qwen27-jubiku.service
