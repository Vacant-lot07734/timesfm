# TimesFM vs Kronos Stage1

This protocol is now owned by the shared workspace `zlab` layer rather than by
the TimesFM repo alone.

## Shared protocol

- Context lengths: `20`, `32`
- Horizons: `1`, `5`
- Target: `close`
- Train: `2025-06-01 ~ 2025-11-30`
- Val: `2025-12-01 ~ 2025-12-31`
- Test: `2026-01-01 ~ 2026-02-28`
- Main metrics: `mean_rank_ic`, `mean_ic`, `mae`

Shared reference:

- `/home/yzh/workspace/zlab/protocol/zero_shot.md`

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
