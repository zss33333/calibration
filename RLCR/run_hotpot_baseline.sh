#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_LABELS=0.50 bash "$ROOT/run_hotpot_weight_sweep.sh"
