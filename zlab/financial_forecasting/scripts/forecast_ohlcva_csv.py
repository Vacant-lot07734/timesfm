#!/usr/bin/env python3
"""Forecast OHLCVA-derived targets with TimesFM 2.5.

TimesFM 2.5 is still a univariate model. This script applies the recommended
zero-shot workflow for financial data:

1. Map each symbol's OHLCVA bars to a single target such as close or returns.
2. Optionally add future-known calendar covariates through XReg.
3. Run either the latest-window forecast or rolling walk-forward backtests.
4. Write flattened forecasts plus evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from timesfm.utils.financial_forecasting import (  # noqa: E402
  OHLCVAColumns,
  apply_experiment_split_mode,
  build_backtest_metrics_payload,
  build_task_collection,
  collate_timesfm_inputs,
  filter_tasks_by_eval_range,
  flatten_predictions,
  load_ohlcva_input,
  materialize_run_layout,
  target_is_nonnegative,
)


def run_preflight() -> dict:
  """Runs the bundled system preflight checks."""

  sys.path.insert(0, str(REPO_ROOT / "timesfm-forecasting" / "scripts"))
  from check_system import run_checks

  report = run_checks("v2.5")
  if not report.passed:
    print("\nSystem check failed. Cannot proceed with forecasting.")
    print(report.verdict_detail)
    print("\nRun 'python timesfm-forecasting/scripts/check_system.py' for details.")
    sys.exit(1)
  return report.to_dict()


def load_model(
  model_id: str,
  batch_size: int,
  context_length: int,
  horizon: int,
  target: str,
  use_xreg: bool,
):
  """Loads and compiles the TimesFM 2.5 model."""

  import torch
  import timesfm

  torch.set_float32_matmul_precision("high")

  print(f"Loading TimesFM 2.5 checkpoint: {model_id}")
  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)

  use_continuous_quantile_head = horizon <= 1024
  if not use_continuous_quantile_head:
    print(
      "Horizon exceeds 1024. Falling back to the default quantile head for this run."
    )
  # TODO 这里的compile是在做什么？
  model.compile(
    timesfm.ForecastConfig(
      max_context=context_length,
      max_horizon=horizon,
      normalize_inputs=True,
      per_core_batch_size=batch_size,
      use_continuous_quantile_head=use_continuous_quantile_head,
      force_flip_invariance=True,
      infer_is_positive=target_is_nonnegative(target),
      fix_quantile_crossing=True,
      return_backcast=use_xreg,
    )
  )
  return model


def run_forecast(model, tasks, use_xreg: bool, xreg_mode: str):
  """Runs TimesFM on a prepared task batch."""

  payload = collate_timesfm_inputs(tasks)
  horizon = len(tasks[0].future_timestamps)

  if use_xreg:
    try:
      point_forecast, quantile_forecast = model.forecast_with_covariates(
        inputs=payload["inputs"],
        dynamic_numerical_covariates=payload["dynamic_numerical_covariates"],
        dynamic_categorical_covariates=payload["dynamic_categorical_covariates"],
        static_numerical_covariates=payload["static_numerical_covariates"],
        static_categorical_covariates=payload["static_categorical_covariates"],
        xreg_mode=xreg_mode,
      )
    except ImportError as exc:
      raise RuntimeError(
        "XReg dependencies are missing. Install them with `uv pip install -e .[xreg]` "
        "or `pip install timesfm[xreg]`."
      ) from exc
  else:
    point_forecast, quantile_forecast = model.forecast(
      horizon=horizon,
      inputs=payload["inputs"],
    )
  return point_forecast, quantile_forecast


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Forecast OHLCVA-derived financial targets with TimesFM 2.5.",
  )
  parser.add_argument(
    "input",
    help="Path to a single OHLCVA CSV file or a directory of per-symbol CSV files.",
  )
  parser.add_argument("--output", default="ohlcva_forecasts.csv", help="Output CSV path.")
  parser.add_argument(
    "--metrics-output",
    default=None,
    help="Optional JSON path for evaluation metrics. Auto-derived in backtest mode.",
  )
  parser.add_argument(
    "--daily-metrics-output",
    default=None,
    help="Optional CSV path for daily cross-sectional metrics in backtest mode.",
  )
  parser.add_argument(
    "--mode",
    choices=["forecast", "backtest"],
    default="backtest",
    help="Use the latest context window or rolling walk-forward backtests.",
  )
  parser.add_argument(
    "--target",
    choices=[
      "close",
      "log_close",
      "close_return",
      "log_close_return",
      "hlc3",
      "ohlc4",
      "volume",
      "log_volume",
      "vwap",
    ],
    default="log_close_return",
    help="Single target derived from OHLCVA bars.",
  )
  parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon.")
  parser.add_argument(
    "--context-length",
    type=int,
    default=512,
    help="Maximum context length passed to TimesFM.",
  )
  parser.add_argument(
    "--min-context",
    type=int,
    default=128,
    help="Minimum history required per symbol after target preprocessing.",
  )
  parser.add_argument(
    "--stride",
    type=int,
    default=None,
    help="Backtest stride. Defaults to the horizon.",
  )
  parser.add_argument(
    "--symbols",
    default=None,
    help="Optional comma-separated symbol whitelist.",
  )
  parser.add_argument(
    "--model-id",
    default="google/timesfm-2.5-200m-pytorch",
    help="Hugging Face checkpoint or local model directory.",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help="Override per_core_batch_size. Auto-detected from preflight if omitted.",
  )
  parser.add_argument(
    "--skip-check",
    action="store_true",
    help="Skip the system preflight check.",
  )
  parser.add_argument(
    "--no-xreg",
    action="store_true",
    help="Disable future-known calendar covariates and use plain TimesFM forecast().",
  )
  parser.add_argument(
    "--xreg-mode",
    choices=["xreg + timesfm", "timesfm + xreg"],
    default="xreg + timesfm",
    help="How TimesFM and XReg are composed when covariates are enabled.",
  )
  parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column.")
  parser.add_argument("--symbol-col", default="symbol", help="Symbol column.")
  parser.add_argument("--open-col", default="open", help="Open column.")
  parser.add_argument("--high-col", default="high", help="High column.")
  parser.add_argument("--low-col", default="low", help="Low column.")
  parser.add_argument("--close-col", default="close", help="Close column.")
  parser.add_argument("--volume-col", default="volume", help="Volume column.")
  parser.add_argument(
    "--amount-col",
    default="amount",
    help="Amount column. Use --no-amount if unavailable.",
  )
  parser.add_argument(
    "--no-amount",
    action="store_true",
    help="Unset the amount column when your bars do not contain traded amount.",
  )
  parser.add_argument(
    "--exchange-col",
    default=None,
    help="Optional exchange or venue identifier column.",
  )
  parser.add_argument(
    "--glob",
    default="*.csv",
    help="When input is a directory, glob pattern used to discover CSV files.",
  )
  parser.add_argument(
    "--train-start",
    default="2025-06-01",
    help="Inclusive train split start date based on context_end_date.",
  )
  parser.add_argument(
    "--train-end",
    default="2025-11-30",
    help="Inclusive train split end date based on context_end_date.",
  )
  parser.add_argument(
    "--val-start",
    default="2025-12-01",
    help="Inclusive validation split start date based on context_end_date.",
  )
  parser.add_argument(
    "--val-end",
    default="2025-12-31",
    help="Inclusive validation split end date based on context_end_date.",
  )
  parser.add_argument(
    "--test-start",
    default="2026-01-01",
    help="Inclusive test split start date based on context_end_date.",
  )
  parser.add_argument(
    "--test-end",
    default="2026-02-28",
    help="Inclusive test split end date based on context_end_date.",
  )
  parser.add_argument(
    "--top-k",
    type=int,
    default=20,
    help="Top-k size for cross-sectional return diagnostics.",
  )
  parser.add_argument(
    "--split-mode",
    choices=["protocol", "all"],
    default="protocol",
    help=(
      "Use the train/val/test protocol split, or mark the selected evaluation "
      "range as a single 'all' split."
    ),
  )
  parser.add_argument(
    "--eval-start",
    default=None,
    help="Optional inclusive evaluation start date based on context_end_date.",
  )
  parser.add_argument(
    "--eval-end",
    default=None,
    help="Optional inclusive evaluation end date based on context_end_date.",
  )
  parser.add_argument(
    "--zero-shot-run-dir",
    default=None,
    dest="zero_shot_run_dir",
    help="Optional zero-shot run directory. When set, writes run_config and split subdirs.",
  )
  parser.add_argument(
    "--stage1-run-dir",
    default=None,
    dest="zero_shot_run_dir",
    help=argparse.SUPPRESS,
  )
  parser.add_argument(
    "--run-name",
    default=None,
    help="Optional experiment run name used when materializing zero-shot layout.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if not args.skip_check:
    print("Running system preflight check...")
    report = run_preflight()
    batch_size = args.batch_size or report.get("recommended_batch_size", 32)
  else:
    print("Skipping system preflight check.")
    batch_size = args.batch_size or 32

  loaded = load_ohlcva_input(args.input, glob_pattern=args.glob)
  df = loaded.frame
  print(f"Loaded {len(loaded.source_files)} file(s) with {len(df)} total rows.")
  if loaded.skipped_files:
    print("Skipped files:")
    for path, reason in sorted(loaded.skipped_files.items()):
      print(f"  - {path}: {reason}")
  if args.symbols:
    allowed = {symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()}
    df = df[df[args.symbol_col].astype(str).isin(allowed)].copy()

  columns = OHLCVAColumns(
    time=args.timestamp_col,
    symbol=args.symbol_col,
    open=args.open_col,
    high=args.high_col,
    low=args.low_col,
    close=args.close_col,
    volume=args.volume_col,
    amount=None if args.no_amount else args.amount_col,
    exchange=args.exchange_col,
  )

  prepared = build_task_collection(
    df=df,
    columns=columns,
    target=args.target,
    horizon=args.horizon,
    context_length=args.context_length,
    mode=args.mode,
    stride=args.stride,
    min_context=args.min_context,
  )

  if prepared.skipped_symbols:
    print("Skipped symbols:")
    for symbol, reason in prepared.skipped_symbols.items():
      print(f"  - {symbol}: {reason}")

  if not prepared.tasks:
    raise RuntimeError("No forecast tasks were created. Check your data and arguments.")

  prepared = filter_tasks_by_eval_range(
    prepared,
    eval_start=args.eval_start,
    eval_end=args.eval_end,
  )
  tasks = prepared.tasks

  if not tasks:
    raise RuntimeError("No forecast tasks remain after applying the evaluation date filter.")

  print(
    f"Prepared {len(tasks)} task(s) across "
    f"{len({task.symbol for task in tasks})} symbol(s)."
  )
  # TODO 这里的 xreg 是什么？
  use_xreg = not args.no_xreg
  model = load_model(
    model_id=args.model_id,
    batch_size=batch_size,
    context_length=args.context_length,
    horizon=args.horizon,
    target=args.target,
    use_xreg=use_xreg,
  )

  point_forecast, quantile_forecast = run_forecast(
    model=model,
    tasks=tasks,
    use_xreg=use_xreg,
    xreg_mode=args.xreg_mode,
  )

  forecast_frame = flatten_predictions(
    tasks=tasks,
    point_forecast=point_forecast,
    quantile_forecast=quantile_forecast,
  )

  if args.mode == "backtest":
    forecast_frame = apply_experiment_split_mode(
      forecast_frame,
      split_mode=args.split_mode,
      train_start=args.train_start,
      train_end=args.train_end,
      val_start=args.val_start,
      val_end=args.val_end,
      test_start=args.test_start,
      test_end=args.test_end,
    )

  forecast_path = Path(args.output)
  forecast_path.parent.mkdir(parents=True, exist_ok=True)
  forecast_frame.to_csv(forecast_path, index=False)
  print(f"Wrote {len(forecast_frame)} forecast rows to {forecast_path}")

  if args.mode == "backtest":
    split_names = ["val", "test"] if args.split_mode == "protocol" else ["all"]
    should_write_split_artifacts = not (
      args.split_mode == "all" and split_names == ["all"]
    )
    for split_name in split_names:
      split_frame = forecast_frame[forecast_frame["split"] == split_name].copy()
      if not should_write_split_artifacts:
        continue
      split_path = forecast_path.with_name(
        f"{forecast_path.stem}_{split_name}{forecast_path.suffix}"
      )
      split_frame.to_csv(split_path, index=False)
      print(f"Wrote {len(split_frame)} {split_name} rows to {split_path}")

  should_write_metrics = args.mode == "backtest"
  metrics_payload: dict | None = None
  daily_metrics: pd.DataFrame | None = None
  daily_metrics_path: Path | None = None
  metrics_path: Path | None = None
  if should_write_metrics:
    metrics_payload, daily_metrics = build_backtest_metrics_payload(
      tasks=tasks,
      point_forecast=point_forecast,
      quantile_forecast=quantile_forecast,
      forecast_frame=forecast_frame,
      target=args.target,
      split_mode=args.split_mode,
      eval_start=args.eval_start,
      eval_end=args.eval_end,
      train_start=args.train_start,
      train_end=args.train_end,
      val_start=args.val_start,
      val_end=args.val_end,
      test_start=args.test_start,
      test_end=args.test_end,
      top_k=args.top_k,
      splits=tuple(split_names),
    )

    if daily_metrics is not None:
      daily_metrics_path = Path(
        args.daily_metrics_output
        or forecast_path.with_name(f"{forecast_path.stem}.daily_metrics.csv")
      )
      daily_metrics_path.parent.mkdir(parents=True, exist_ok=True)
      daily_metrics.to_csv(daily_metrics_path, index=False)
      print(f"Wrote daily cross-sectional metrics to {daily_metrics_path}")
    else:
      print(
        "Skipping cross-sectional return metrics because they are only defined "
        "for target=close in the current experiment protocol."
      )

    metrics_path = Path(args.metrics_output or forecast_path.with_suffix(".metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Wrote evaluation metrics to {metrics_path}")

  if args.zero_shot_run_dir:
    if args.mode != "backtest":
      raise ValueError("--zero-shot-run-dir only supports backtest mode.")
    if metrics_payload is None:
      raise ValueError("Zero-shot layout requires metrics output in backtest mode.")
    materialize_run_layout(
      run_dir=args.zero_shot_run_dir,
      forecast_frame=forecast_frame,
      daily_metrics_frame=daily_metrics,
      metrics_payload=metrics_payload,
      run_name=args.run_name or Path(args.zero_shot_run_dir).name,
      model_id=args.model_id,
      target=args.target,
      context_length=args.context_length,
      horizon=args.horizon,
      top_k=args.top_k,
      xreg="disabled" if args.no_xreg else "enabled",
      source_predictions_path=forecast_path,
      source_daily_metrics_path=daily_metrics_path,
      source_metrics_path=metrics_path,
      splits=tuple(split_names),
    )
    print(f"Materialized zero-shot layout to {args.zero_shot_run_dir}")


if __name__ == "__main__":
  main()
