from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from timesfm.utils.financial_forecasting import (
  OHLCVAColumns,
  apply_experiment_split_mode,
  add_experiment_splits,
  build_backtest_metrics_payload,
  build_task_collection,
  build_last_window_tasks,
  build_walk_forward_tasks,
  compute_cross_sectional_metrics,
  collate_timesfm_inputs,
  compute_forecast_metrics,
  enforce_ohlc_constraints,
  filter_tasks_by_eval_range,
  flatten_forecast_frame,
  flatten_predictions,
  load_ohlcva_input,
  make_target_series,
  materialize_run_layout,
  materialize_zero_shot_layout,
  validate_ohlcva_frame,
)


def _make_frame() -> pd.DataFrame:
  rows = []
  timestamps = pd.date_range("2025-01-01", periods=10, freq="B")
  for symbol, base in (("AAA", 100.0), ("BBB", 200.0)):
    for idx, ts in enumerate(timestamps):
      close = base + idx
      open_ = close - 0.5
      high = close + 1.0
      low = close - 1.5
      volume = 1_000 + idx * 10
      rows.append(
        {
          "timestamp": ts,
          "symbol": symbol,
          "exchange": "XNAS",
          "open": open_,
          "high": high,
          "low": low,
          "close": close,
          "volume": volume,
          "amount": close * volume,
        }
      )
  return pd.DataFrame(rows[::-1])


def test_validate_ohlcva_frame_sorts_and_checks_schema():
  frame = _make_frame()
  validated = validate_ohlcva_frame(
    frame,
    OHLCVAColumns(exchange="exchange"),
  )

  assert list(validated.columns) == list(frame.columns)
  assert validated.iloc[0]["symbol"] == "AAA"
  assert validated.iloc[0]["timestamp"] == pd.Timestamp("2025-01-01")
  assert validated.iloc[-1]["symbol"] == "BBB"


def test_make_target_series_generates_log_returns():
  frame = validate_ohlcva_frame(_make_frame(), OHLCVAColumns(exchange="exchange"))
  target = make_target_series(
    frame[frame["symbol"] == "AAA"],
    OHLCVAColumns(exchange="exchange"),
    "log_close_return",
  )

  assert len(target) == 9
  expected = np.log(101.0) - np.log(100.0)
  assert np.isclose(target.iloc[0]["target"], expected)


def test_load_ohlcva_input_normalizes_tushare_style_columns(tmp_path):
  stock_path = tmp_path / "600519.SH_qfq_day.csv"
  metadata_path = tmp_path / "_stock_list_hourly.csv"

  stock_path.write_text(
    "\n".join(
      [
        "ts_code,trade_date,open,high,low,close,vol,amount",
        "600519.SH,20250603,1454.27,1464.87,1451.39,1455.22,28979.18,4377892.799",
        "600519.SH,20250604,1456.67,1468.73,1455.44,1456.15,23008.09,3480363.797",
      ]
    ),
    encoding="utf-8",
  )
  metadata_path.write_text("ts_code,trade_date\n600519.SH,20260227\n", encoding="utf-8")

  loaded = load_ohlcva_input(tmp_path)
  validated = validate_ohlcva_frame(loaded.frame, OHLCVAColumns())

  assert loaded.source_files == [str(stock_path)]
  assert str(metadata_path) in loaded.skipped_files
  assert list(validated["symbol"].unique()) == ["600519.SH"]
  assert validated["timestamp"].iloc[0] == pd.Timestamp("2025-06-03")
  assert "volume" in validated.columns


def test_build_walk_forward_tasks_generates_calendar_covariates():
  prepared = build_walk_forward_tasks(
    df=_make_frame(),
    columns=OHLCVAColumns(exchange="exchange"),
    target="close",
    horizon=2,
    context_length=4,
    min_context=4,
  )

  assert prepared.skipped_symbols == {}
  assert len(prepared.tasks) == 10

  first_task = prepared.tasks[0]
  assert first_task.symbol == "AAA"
  assert len(first_task.context_values) == 4
  assert len(first_task.future_timestamps) == 2
  assert len(first_task.dynamic_numerical_covariates["time_index"]) == 6
  assert len(first_task.dynamic_categorical_covariates["day_of_week"]) == 6

  payload = collate_timesfm_inputs(prepared.tasks[:2])
  assert len(payload["inputs"]) == 2
  assert len(payload["dynamic_numerical_covariates"]["time_index"]) == 2
  assert len(payload["static_categorical_covariates"]["symbol"]) == 2


def test_build_task_collection_and_eval_range_filter_wrapper():
  prepared = build_task_collection(
    df=_make_frame(),
    columns=OHLCVAColumns(exchange="exchange"),
    target="close",
    horizon=2,
    context_length=4,
    mode="backtest",
    min_context=4,
  )

  assert len(prepared.tasks) == 10

  filtered = filter_tasks_by_eval_range(
    prepared,
    eval_start="2025-01-09",
    eval_end="2025-01-10",
  )

  assert len(filtered.tasks) == 2
  assert {
    task.future_timestamps[0].strftime("%Y-%m-%d")
    for task in filtered.tasks
  } == {"2025-01-09"}
  assert {
    task.future_timestamps[-1].strftime("%Y-%m-%d")
    for task in filtered.tasks
  } == {"2025-01-10"}


def test_filter_tasks_by_eval_range_rejects_horizon_spillover():
  prepared = build_task_collection(
    df=_make_frame(),
    columns=OHLCVAColumns(exchange="exchange"),
    target="close",
    horizon=2,
    context_length=4,
    mode="backtest",
    min_context=4,
  )

  filtered = filter_tasks_by_eval_range(
    prepared,
    eval_start="2025-01-09",
    eval_end="2025-01-09",
  )

  assert len(filtered.tasks) == 0


def test_build_last_window_tasks_skips_short_symbols():
  frame = _make_frame()
  short = frame[frame["symbol"] == "AAA"].head(3).copy()
  long = frame[frame["symbol"] == "BBB"].copy()
  mixed = pd.concat([short, long], ignore_index=True)

  prepared = build_last_window_tasks(
    df=mixed,
    columns=OHLCVAColumns(exchange="exchange"),
    target="close",
    horizon=2,
    context_length=5,
    min_context=4,
  )

  assert len(prepared.tasks) == 1
  assert prepared.tasks[0].symbol == "BBB"
  assert "AAA" in prepared.skipped_symbols


def test_compute_forecast_metrics_and_constraint_postprocessing():
  prepared = build_walk_forward_tasks(
    df=_make_frame(),
    columns=OHLCVAColumns(exchange="exchange"),
    target="close",
    horizon=2,
    context_length=4,
    min_context=4,
  )
  tasks = prepared.tasks[:2]
  actual = np.stack([task.future_values for task in tasks], axis=0)
  point = actual.copy()

  metrics = compute_forecast_metrics(tasks, point)
  assert metrics["mae"] == 0.0
  assert metrics["rmse"] == 0.0
  assert metrics["da"] == 1.0

  constrained = enforce_ohlc_constraints(
    {
      "open": np.array([10.0, 11.0]),
      "high": np.array([9.0, 10.5]),
      "low": np.array([10.5, 12.0]),
      "close": np.array([12.0, 9.0]),
      "volume": np.array([-1.0, 3.0]),
    }
  )
  assert np.all(constrained["high"] >= np.maximum(constrained["open"], constrained["close"]))
  assert np.all(constrained["low"] <= np.minimum(constrained["open"], constrained["close"]))
  assert np.all(constrained["volume"] >= 0.0)


def test_flatten_forecast_frame_for_close_includes_return_fields():
  prepared = build_walk_forward_tasks(
    df=_make_frame(),
    columns=OHLCVAColumns(exchange="exchange"),
    target="close",
    horizon=2,
    context_length=4,
    min_context=4,
  )
  task = prepared.tasks[0]
  point = np.array([[105.0, 106.0]], dtype=np.float32)
  quantiles = np.zeros((1, 2, 10), dtype=np.float32)
  quantiles[..., 0] = point
  quantiles[..., 5] = point

  frame = flatten_forecast_frame([task], point, quantiles)

  assert list(frame["instrument"].unique()) == [task.symbol]
  assert list(frame["horizon"]) == [1, 2]
  assert np.allclose(frame["anchor_close"], task.context_values[-1])
  assert np.allclose(frame["pred_close"], [105.0, 106.0])
  assert np.allclose(
    frame["pred_return"],
    np.array([105.0, 106.0], dtype=np.float32) / task.context_values[-1] - 1.0,
  )
  assert list(pd.to_datetime(frame["prediction_start_date"])) == [task.future_timestamps[0]] * 2
  assert list(pd.to_datetime(frame["prediction_end_date"])) == [task.future_timestamps[-1]] * 2
  assert "true_close" in frame.columns
  assert "true_return" in frame.columns


def test_add_experiment_splits_requires_full_horizon_within_split():
  frame = pd.DataFrame(
    {
      "prediction_start_date": pd.to_datetime(["2024-06-24", "2024-06-27", "2024-07-15", "2025-12-29"]),
      "prediction_end_date": pd.to_datetime(["2024-06-28", "2024-07-03", "2024-07-19", "2026-01-05"]),
      "value": [1, 2, 3, 4],
    }
  )

  tagged = add_experiment_splits(frame)

  assert list(tagged["split"]) == ["train", "out_of_range", "val", "out_of_range"]


def test_apply_experiment_split_mode_supports_all():
  frame = pd.DataFrame(
    {
      "prediction_start_date": pd.to_datetime(["2025-11-28", "2025-12-15"]),
      "value": [1, 2],
    }
  )

  tagged = apply_experiment_split_mode(frame, split_mode="all")

  assert list(tagged["split"]) == ["all", "all"]


def test_compute_cross_sectional_metrics_groups_by_day_and_horizon():
  frame = pd.DataFrame(
    {
      "instrument": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
      "prediction_start_date": pd.to_datetime(
        [
          "2025-12-01",
          "2025-12-01",
          "2025-12-01",
          "2025-12-02",
          "2025-12-02",
          "2025-12-02",
        ]
      ),
      "prediction_end_date": pd.to_datetime(
        [
          "2025-12-02",
          "2025-12-02",
          "2025-12-02",
          "2025-12-03",
          "2025-12-03",
          "2025-12-03",
        ]
      ),
      "horizon": [1, 1, 1, 1, 1, 1],
      "pred_return": [0.30, 0.20, 0.10, 0.30, 0.20, 0.10],
      "true_return": [0.03, 0.02, 0.01, 0.01, 0.02, 0.03],
      "pred_close": [130.0, 120.0, 110.0, 130.0, 120.0, 110.0],
      "true_close": [103.0, 102.0, 101.0, 101.0, 102.0, 103.0],
      "split": ["val", "val", "val", "val", "val", "val"],
    }
  )

  daily_metrics, summary = compute_cross_sectional_metrics(frame)

  assert len(daily_metrics) == 2
  assert np.isclose(daily_metrics.iloc[0]["rank_ic"], 1.0)
  assert np.isclose(daily_metrics.iloc[0]["ic"], 1.0)
  assert np.isclose(daily_metrics.iloc[1]["rank_ic"], -1.0)
  assert np.isclose(daily_metrics.iloc[1]["ic"], -1.0)

  assert len(summary["rows"]) == 1
  row = summary["rows"][0]
  assert row["split"] == "val"
  assert row["horizon"] == 1
  assert np.isclose(row["rank_ic"], 0.0)
  assert np.isclose(row["ic"], 0.0)
  assert np.isclose(row["da"], 1.0)
  assert np.isclose(row["mae"], 0.18)
  assert np.isclose(row["rmse"], np.sqrt((0.27**2 + 0.18**2 + 0.09**2 + 0.29**2 + 0.18**2 + 0.07**2) / 6.0))
  assert np.isclose(row["rank_icir"], 0.0)
  assert np.isclose(row["icir"], 0.0)


def test_materialize_zero_shot_layout_writes_expected_split_files(tmp_path):
  forecast_frame = pd.DataFrame(
    {
      "instrument": ["AAA", "BBB"],
      "context_end_date": pd.to_datetime(["2025-12-01", "2026-01-02"]),
      "prediction_start_date": pd.to_datetime(["2025-12-02", "2026-01-03"]),
      "prediction_end_date": pd.to_datetime(["2025-12-02", "2026-01-07"]),
      "horizon": [1, 1],
      "pred_return": [0.1, 0.2],
      "true_return": [0.05, 0.25],
      "split": ["val", "test"],
    }
  )
  daily_metrics = pd.DataFrame(
    {
      "split": ["val", "test"],
      "horizon": [1, 1],
      "prediction_start_date": pd.to_datetime(["2025-12-02", "2026-01-03"]),
      "ic": [1.0, -1.0],
      "rank_ic": [1.0, -1.0],
    }
  )
  metrics_payload = {
    "experiment_protocol": {"target": "close"},
    "split_summaries": [
      {"split": "val", "horizon": 1, "rank_ic": 1.0, "ic": 1.0, "rank_icir": None, "icir": None, "da": 1.0, "mae": 0.05, "rmse": 0.05},
      {"split": "test", "horizon": 1, "rank_ic": -1.0, "ic": -1.0, "rank_icir": None, "icir": None, "da": 1.0, "mae": 0.05, "rmse": 0.05},
    ],
  }

  materialize_zero_shot_layout(
    run_dir=tmp_path / "run",
    forecast_frame=forecast_frame,
    daily_metrics_frame=daily_metrics,
    metrics_payload=metrics_payload,
    run_name="timesfm_zero_shot_ctx32_q50",
    model_id="google/timesfm-2.5-200m-pytorch",
    target="close",
    context_length=32,
    horizon=1,
    xreg="disabled",
    source_predictions_path=tmp_path / "predictions_all.csv",
    source_daily_metrics_path=tmp_path / "daily_metrics_all.csv",
    source_metrics_path=tmp_path / "metrics_all.json",
  )

  run_dir = tmp_path / "run"
  assert (run_dir / "run_config.json").exists()
  assert (run_dir / "val" / "predictions.csv").exists()
  assert (run_dir / "val" / "daily_metrics.csv").exists()
  assert (run_dir / "val" / "metrics.json").exists()
  assert (run_dir / "test" / "predictions.csv").exists()
  assert (run_dir / "test" / "daily_metrics.csv").exists()
  assert (run_dir / "test" / "metrics.json").exists()

  val_metrics = json.loads((run_dir / "val" / "metrics.json").read_text())
  assert val_metrics["prediction_rows"] == 1
  assert val_metrics["rank_ic"] == 1.0


def test_materialize_zero_shot_layout_supports_all_split(tmp_path):
  forecast_frame = pd.DataFrame(
    {
      "instrument": ["AAA"],
      "context_end_date": pd.to_datetime(["2025-12-01"]),
      "prediction_start_date": pd.to_datetime(["2025-12-02"]),
      "prediction_end_date": pd.to_datetime(["2025-12-02"]),
      "horizon": [1],
      "pred_return": [0.1],
      "true_return": [0.05],
      "split": ["all"],
    }
  )
  daily_metrics = pd.DataFrame(
    {
      "split": ["all"],
      "horizon": [1],
      "prediction_start_date": pd.to_datetime(["2025-12-02"]),
      "ic": [None],
      "rank_ic": [None],
    }
  )

  materialize_zero_shot_layout(
    run_dir=tmp_path / "run",
    forecast_frame=forecast_frame,
    daily_metrics_frame=daily_metrics,
    metrics_payload={"experiment_protocol": {"split_mode": "all"}},
    run_name="timesfm_zero_shot_ctx20_q50_all",
    model_id="google/timesfm-2.5-200m-pytorch",
    target="close",
    context_length=20,
    horizon=1,
    xreg="disabled",
    splits=("all",),
  )

  run_dir = tmp_path / "run"
  assert (run_dir / "all" / "predictions.csv").exists()
  assert (run_dir / "all" / "daily_metrics.csv").exists()
  assert (run_dir / "all" / "metrics.json").exists()


def test_shared_wrappers_build_metrics_and_layout(tmp_path):
  prepared = build_walk_forward_tasks(
    df=_make_frame(),
    columns=OHLCVAColumns(exchange="exchange"),
    target="close",
    horizon=1,
    context_length=4,
    min_context=4,
  )
  tasks = prepared.tasks[:2]
  point = np.stack([task.future_values for task in tasks], axis=0)
  frame = flatten_predictions(tasks, point)
  frame = apply_experiment_split_mode(frame, split_mode="all")

  metrics_payload, daily_metrics = build_backtest_metrics_payload(
    tasks=tasks,
    point_forecast=point,
    quantile_forecast=None,
    forecast_frame=frame,
    target="close",
    split_mode="all",
    eval_start="2025-01-01",
    eval_end="2025-01-31",
    train_start="2021-01-01",
    train_end="2024-06-30",
    val_start="2024-07-01",
    val_end="2024-12-31",
    test_start="2025-01-01",
    test_end="2025-12-31",
    splits=("all",),
  )

  assert daily_metrics is not None
  assert "split_summaries" in metrics_payload

  materialize_run_layout(
    run_dir=tmp_path / "run",
    forecast_frame=frame,
    daily_metrics_frame=daily_metrics,
    metrics_payload=metrics_payload,
    run_name="timesfm_zero_shot_ctx20_q50_all",
    model_id="google/timesfm-2.5-200m-pytorch",
    target="close",
    context_length=20,
    horizon=1,
    xreg="disabled",
    splits=("all",),
  )

  assert (tmp_path / "run" / "all" / "metrics.json").exists()
