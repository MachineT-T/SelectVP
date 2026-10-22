#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="configs/qvhighlights.yaml"
python tools/prepare_data.py --config "$config" --check --annotations-only "$@"
"${SELECTVP_BLIP2_PYTHON:-python}" tools/preprocess.py --config "$config" --stages videos,blip2 "$@"
python tools/preprocess.py --config "$config" --stages events,targets,detection,keyframes,prompts "$@"
python tools/infer.py --config "$config" "$@"
python tools/evaluate.py --config "$config" "$@"
