# TimesFM vs Kronos Zero-Shot

This protocol is now owned by the shared workspace `zlab` layer rather than by
the TimesFM repo alone.

## Shared protocol

- Context lengths: `20`, `32`
- Horizons: `1`, `5`
- Target: `close`
- If a configured range start is not a trading day, use the first trading day
  inside the range as the first `prediction_start_date`
- Strict split rule: keep windows only when `prediction_start_date` is in range
  and `prediction_end_date` stays on or before the split end
- Train: `2021-01-01 ~ 2024-06-30`
- Val: `2024-07-01 ~ 2024-12-31`
- Test: `2025-01-01 ~ 2025-12-31`
- Main metrics: `rank_ic`, `ic`, `rank_icir`, `icir`
- Supporting metrics: `da`, `mae`, `rmse`

Shared reference:

- `/home/yzh/workspace/zlab/protocol/zero_shot.md`
- `/home/yzh/workspace/zlab/protocol/evaluation_metrics.md`
- `/home/yzh/workspace/zlab/protocol/evaluation_metrics.json`
- `/home/yzh/workspace/zlab/protocol/sample_selection.md`

Deferred strategy-validation metrics:

- `AER`
- `IR`

Current status:

- TimesFM and Kronos now follow the shared strict full-horizon split rule
- the remaining adapter-specific caveats are documented in
  `/home/yzh/workspace/zlab/protocol/sample_selection.md`

## Preferred commands

Run Kronos:

```bash
bash /home/yzh/workspace/zlab/scripts/run_kronos_zero_shot.sh
```

Run TimesFM:

```bash
bash /home/yzh/workspace/zlab/scripts/run_timesfm_zero_shot.sh
```

Compare results:

```bash
python /home/yzh/workspace/zlab/scripts/compare_zero_shot.py \
  kronos=/home/yzh/workspace/zlab/results/zero_shot/kronos \
  timesfm=/home/yzh/workspace/zlab/results/zero_shot/timesfm \
  --output-dir /home/yzh/workspace/zlab/results/comparisons
```

## Repo-local compatibility wrappers

TimesFM repo:

```bash
bash zlab/financial_forecasting/scripts/run_timesfm_zero_shot.sh
```

Kronos repo:

```bash
bash zlab/scripts/run_kronos_zero_shot.sh
```

These wrappers keep legacy env names usable, but the shared workspace scripts
are now the canonical zero-shot entrypoints.
