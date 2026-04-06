# TimesFM Financial Forecasting Refactor Note

## Current status

The old assumption that TimesFM still lacked protocol split handling and
`rankIC / IC` output is no longer true. The repo already supports:

- `train / val / test` protocol dates
- `all` versus `protocol` split mode
- cross-sectional `ic / rank_ic`
- zero-shot run layout materialization

The remaining refactor goal is not to re-implement these features again, but to
reorganize ownership so they fit the new shared workspace experiment layer.

## New ownership split

### Workspace-level `/home/yzh/workspace/zlab`

Owns:

- shared experiment defaults
- shared zero-shot run scripts
- shared compare and metric collection scripts
- shared result root

### Repo-level `timesfm/zlab/financial_forecasting`

Owns:

- TimesFM-specific CLI
- OHLCVA loading / task construction
- TimesFM forecast flattening
- cross-sectional metric payload assembly
- TimesFM tests and adapter docs

## Implementation changes in this refactor

### 1. Shared workspace entrypoints

Added:

- `/home/yzh/workspace/zlab/scripts/run_timesfm.sh`
- `/home/yzh/workspace/zlab/scripts/run_kronos.sh`
- `/home/yzh/workspace/zlab/scripts/compare_models.py`
- `/home/yzh/workspace/zlab/scripts/collect_metrics.py`

### 2. TimesFM tool-layer cleanup

`src/timesfm/utils/financial_forecasting.py` now exposes shared-runner-friendly
helpers:

- `build_task_collection(...)`
- `filter_tasks_by_eval_range(...)`
- `apply_experiment_split_mode(...)`
- `build_backtest_metrics_payload(...)`
- `flatten_predictions(...)`
- `materialize_run_layout(...)`

These wrap the existing lower-level helpers instead of replacing them.

### 3. Repo-local wrapper behavior

Legacy `zlab/financial_forecasting/scripts/run_timesfm_stage1.sh` is now a compatibility
wrapper. It maps legacy `TIMESFM_*` env vars into the shared `ZLAB_*` contract
and delegates execution to `/home/yzh/workspace/zlab/scripts/run_timesfm_zero_shot.sh`.

## Immediate usage

Preferred:

```bash
bash /home/yzh/workspace/zlab/scripts/run_timesfm_zero_shot.sh
```

Compatible:

```bash
bash zlab/financial_forecasting/scripts/run_timesfm_zero_shot.sh
```
