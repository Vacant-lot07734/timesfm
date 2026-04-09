# TimesFM Financial Forecasting Adapter

`timesfm/zlab/financial_forecasting/` now serves as the TimesFM-side adapter
layer for the shared workspace experiment stack:

- shared cross-model entrypoint: `/home/yzh/workspace/zlab`
- repo-local implementation and tests: `timesfm/zlab/financial_forecasting`

## Responsibilities

This directory keeps:

- the TimesFM execution entrypoint
- the OHLCVA-to-TimesFM task conversion helpers
- TimesFM-specific tests and notes

It does not own the cross-model orchestration anymore. Shared zero-shot runs and
comparisons now live under the workspace-level `zlab/`.

## Preferred entrypoints

Run the shared workspace script directly:

```bash
bash /home/yzh/workspace/zlab/scripts/run_timesfm_zero_shot.sh
```

Or use the repo-local compatibility wrapper:

```bash
bash zlab/financial_forecasting/scripts/run_timesfm_zero_shot.sh
```

The compatibility wrapper forwards legacy `TIMESFM_*` environment variables into
the shared `ZLAB_*` contract and then delegates to the workspace script.

## Repo-local execution entry

The low-level runner remains:

```bash
python zlab/financial_forecasting/scripts/forecast_ohlcva_csv.py \
  /home/yzh/workspace/Kronos-0/zlab/data/daily \
  --mode backtest \
  --horizon 5 \
  --context-length 32 \
  --split-mode all \
  --no-xreg
```

This script is still the right place for TimesFM-specific changes such as:

- task construction
- eval-range filtering
- return-based metric payload assembly
- zero-shot run layout materialization

## Related docs

- shared workspace protocol: `/home/yzh/workspace/zlab/protocol/zero_shot.md`
- shared comparison script: `/home/yzh/workspace/zlab/scripts/compare_models.py`
- current repo-side design note: `zlab/financial_forecasting/DEV_PLAN.md`
