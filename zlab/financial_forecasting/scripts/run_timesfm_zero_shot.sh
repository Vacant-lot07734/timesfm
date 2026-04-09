#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
SHARED_ZLAB_ROOT="${ZLAB_WORKSPACE_ROOT:-$WORKSPACE_ROOT/zlab}"

if [[ ! -f "$SHARED_ZLAB_ROOT/env/defaults.sh" ]]; then
  echo "Shared workspace zlab was not found at: $SHARED_ZLAB_ROOT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$SHARED_ZLAB_ROOT/env/defaults.sh"

export ZLAB_TIMESFM_REPO="$REPO_ROOT"
export ZLAB_TIMESFM_PYTHON="${TIMESFM_PYTHON:-$ZLAB_TIMESFM_PYTHON}"
export ZLAB_DATA_ROOT="${TIMESFM_DATA_DIR:-$ZLAB_DATA_ROOT}"
export ZLAB_TIMESFM_MODEL_ID="${TIMESFM_MODEL_ID:-$ZLAB_TIMESFM_MODEL_ID}"
export ZLAB_CONTEXTS="${TIMESFM_CONTEXTS:-$ZLAB_CONTEXTS}"
export ZLAB_TIMESFM_BATCH_SIZE="${TIMESFM_BATCH_SIZE:-$ZLAB_TIMESFM_BATCH_SIZE}"
export ZLAB_TIMESFM_RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-$ZLAB_TIMESFM_RUN_NAME_PREFIX}"
export ZLAB_TIMESFM_ZERO_SHOT_ROOT="${ZERO_SHOT_ROOT:-${STAGE1_ROOT:-$ZLAB_TIMESFM_ZERO_SHOT_ROOT}}"
export ZLAB_SPLIT_MODE="${TIMESFM_SPLIT_MODE:-$ZLAB_SPLIT_MODE}"
export ZLAB_EVAL_START="${TIMESFM_EVAL_START:-${ZLAB_EVAL_START:-}}"
export ZLAB_EVAL_END="${TIMESFM_EVAL_END:-${ZLAB_EVAL_END:-}}"

if [[ -n "${TIMESFM_SYMBOLS:-}" && -z "${ZLAB_TIMESFM_SYMBOLS:-}" ]]; then
  export ZLAB_TIMESFM_SYMBOLS="$TIMESFM_SYMBOLS"
fi
if [[ -n "${TIMESFM_USE_XREG:-}" && -z "${ZLAB_TIMESFM_USE_XREG:-}" ]]; then
  export ZLAB_TIMESFM_USE_XREG="$TIMESFM_USE_XREG"
fi
if [[ -n "${TIMESFM_XREG_MODE:-}" && -z "${ZLAB_TIMESFM_XREG_MODE:-}" ]]; then
  export ZLAB_TIMESFM_XREG_MODE="$TIMESFM_XREG_MODE"
fi

CMD=("$SHARED_ZLAB_ROOT/scripts/run_timesfm_zero_shot.sh")
if [[ -n "${ZLAB_TIMESFM_SYMBOLS:-}" ]]; then
  CMD+=(--symbols "$ZLAB_TIMESFM_SYMBOLS")
fi
if [[ "${ZLAB_TIMESFM_USE_XREG:-false}" == "true" ]]; then
  CMD+=(--with-xreg --xreg-mode "${ZLAB_TIMESFM_XREG_MODE:-xreg + timesfm}")
fi
if [[ -n "${ZLAB_EVAL_START:-}" ]]; then
  CMD+=(--eval-start "$ZLAB_EVAL_START")
fi
if [[ -n "${ZLAB_EVAL_END:-}" ]]; then
  CMD+=(--eval-end "$ZLAB_EVAL_END")
fi

"${CMD[@]}" "$@"
