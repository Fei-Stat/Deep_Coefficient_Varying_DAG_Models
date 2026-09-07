"""Prepare frozen Reddit features and contemporaneous 99-stock log returns.

No torch dependency is needed to prepare the data. Input workbooks are read only.
Run ``python news_stock_data.py --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl

from news_stock_config import EXCLUDED_STOCKS, EXPECTED_STOCK_COUNT, NEWS_SHAPE


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date_string(value):
    if isinstance(value, dt.datetime):
        if value.time() != dt.time():
            raise ValueError("Expected daily dates, found a non-midnight timestamp.")
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value.strip())
        if parsed.time() != dt.time() or parsed.tzinfo is not None:
            raise ValueError("Expected daily dates without time zones.")
        return parsed.date().isoformat()
    raise ValueError(f"Invalid date: {value!r}")


def read_daily_matrix(path, sheet_name=None):
    """Read Date + numeric columns, checking headers, dates, and cell types."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[sheet_name] if sheet_name else workbook.worksheets[0]
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator, None)
        if header is None or header[0] != "Date":
            raise ValueError(f"{path}: first column must be Date.")
        columns = [str(item).strip() if item is not None else "" for item in header[1:]]
        if not columns or any(not c for c in columns) or len(set(columns)) != len(columns):
            raise ValueError(f"{path}: empty or duplicate numeric column names.")
        dates, values = [], []
        for row_number, row in enumerate(iterator, start=2):
            if all(value is None for value in row):
                continue
            dates.append(_date_string(row[0]))
            numeric = []
            for column, value in zip(columns, row[1:]):
                if value is None:
                    numeric.append(np.nan)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.append(float(value))
                else:
                    raise ValueError(f"{path}, row {row_number}, {column}: not numeric.")
            values.append(numeric)
    finally:
        workbook.close()
    if not dates or len(set(dates)) != len(dates):
        raise ValueError(f"{path}: empty data or duplicate dates.")
    order = np.argsort(dates)
    return np.asarray(dates)[order], columns, np.asarray(values, dtype=np.float64)[order]


def read_news(path, sheet_name=None):
    dates, columns, matrix = read_daily_matrix(path, sheet_name)
    expected = [f"T{t:02d}_dim{k}" for t in range(1, 26) for k in range(1, 51)]
    if set(columns) != set(expected) or len(columns) != len(expected):
        raise ValueError("News must contain exactly T01_dim1 through T25_dim50.")
    positions = {name: index for index, name in enumerate(columns)}
    matrix = matrix[:, [positions[name] for name in expected]]
    if not np.isfinite(matrix).all():
        raise ValueError("News contains missing/non-finite values; zero vectors are allowed.")
    result = matrix.astype(np.float32).reshape(-1, *NEWS_SHAPE)
    if not np.isfinite(result).all():
        raise ValueError("News features exceed float32 range.")
    return dates, result


def prepare_data(prices_path, news_path, output, train_fraction=0.7,
                 val_fraction=0.15, price_sheet=None, news_sheet=None):
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Prepared output must have the .npz extension.")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Prepared output already exists; choose another output path.")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1
            and train_fraction + val_fraction < 1):
        raise ValueError("Train/validation fractions must be positive and sum to less than 1.")

    price_dates, tickers, prices = read_daily_matrix(prices_path, price_sheet)
    keep = [i for i, ticker in enumerate(tickers) if ticker not in EXCLUDED_STOCKS]
    stock_ids = np.asarray([tickers[i] for i in keep])
    if len(stock_ids) != EXPECTED_STOCK_COUNT:
        raise ValueError(f"Expected 99 retained stocks, found {len(stock_ids)}. "
                         "Use the supplied 104-stock workbook and fixed exclusion list.")
    prices = prices[:, keep]
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Retained stock prices must be complete, finite, and positive. "
                         "No filling or automatic date deletion is performed.")
    # Crucial: diff BEFORE the news join, so a missing news day cannot create
    # a multi-session return labeled as a single-session return.
    log_returns = np.diff(np.log(prices), axis=0)
    news_dates, news = read_news(news_path, news_sheet)
    news_lookup = {date: i for i, date in enumerate(news_dates)}
    return_rows = [i for i, date in enumerate(price_dates[1:]) if date in news_lookup]
    dates = price_dates[1:][return_rows]
    previous_dates = price_dates[:-1][return_rows]
    returns = log_returns[return_rows]
    features = news[[news_lookup[date] for date in dates]]
    n = len(dates)
    n_train, n_val = int(n * train_fraction), int(n * val_fraction)
    if min(n_train, n_val, n - n_train - n_val) < 2:
        raise ValueError("Each chronological split must contain at least two samples.")
    indices = {
        "train": np.arange(n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, n),
    }
    mean = returns[indices["train"]].mean(axis=0)
    scale = returns[indices["train"]].std(axis=0, ddof=0)
    if np.any(scale < 1e-8):
        bad = stock_ids[scale < 1e-8].tolist()
        raise ValueError(f"Near-constant training returns require review: {bad}")
    y = ((returns - mean) / scale).astype(np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite standardized returns.")
    metadata = {
        "format_version": 1,
        "task": "same_day_news_conditioned_stock_structure",
        "return_definition": "log(P_t)-log(P_previous_price_session)",
        "price_adjustment": "used_as_supplied; split-adjustment evidence; dividend status unverified",
        "excluded_stocks": [ticker for ticker in EXCLUDED_STOCKS if ticker in tickers],
        "stock_ids": stock_ids.tolist(),
        "n_samples": n,
        "n_stocks": len(stock_ids),
        "news_shape": list(NEWS_SHAPE),
        "news_preprocessing": "frozen supplied features, raw zeros retained, no feature scaling",
        "news_zero_blocks_all_input": int(np.all(news == 0, axis=2).sum()),
        "news_zero_blocks_paired": int(np.all(features == 0, axis=2).sum()),
        "price_dates_without_news": sorted(set(price_dates) - set(news_dates)),
        "news_dates_without_return": sorted(set(news_dates) - set(price_dates[1:])),
        "standardization": "per-stock, training mean/std only, ddof=0, no clipping",
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "splits": {name: {"n": len(idx), "start": str(dates[idx[0]]),
                          "end": str(dates[idx[-1]])} for name, idx in indices.items()},
        "sources": {"prices": {"filename": Path(prices_path).name,
                               "sha256": file_sha256(prices_path)},
                    "news": {"filename": Path(news_path).name,
                             "sha256": file_sha256(news_path)}},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, news=features, Y=y, log_returns=returns,
                        dates=dates, previous_dates=previous_dates, stock_ids=stock_ids,
                        mean=mean, scale=scale,
                        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                        **{f"{name}_indices": idx for name, idx in indices.items()})
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return metadata


def load_prepared(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data["metadata"] = json.loads(str(data.pop("metadata_json")))
    if data["metadata"].get("format_version") != 1:
        raise ValueError("Unsupported prepared-data version.")
    n, p = data["Y"].shape
    if data["news"].shape != (n, *NEWS_SHAPE) or len(data["stock_ids"]) != p:
        raise ValueError("Invalid news/return shapes.")
    if data["log_returns"].shape != (n, p) or data["mean"].shape != (p,) or data["scale"].shape != (p,):
        raise ValueError("Invalid return/scaler shapes.")
    for key in ("news", "Y", "log_returns", "mean", "scale"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Non-finite prepared array: {key}")
    if np.any(data["scale"] <= 0):
        raise ValueError("Non-positive scales.")
    if len(data["dates"]) != n or len(set(data["dates"])) != n or np.any(data["dates"][1:] <= data["dates"][:-1]):
        raise ValueError("Dates must be unique and strictly chronological.")
    parts = [data[f"{split}_indices"] for split in ("train", "val", "test")]
    if any(part.ndim != 1 or len(part) < 2 or part.dtype.kind not in "iu" for part in parts):
        raise ValueError("Invalid split indices.")
    if not np.array_equal(np.concatenate(parts), np.arange(n)):
        raise ValueError("Splits must partition the samples chronologically without overlap.")
    if not np.allclose(data["Y"], (data["log_returns"] - data["mean"]) / data["scale"], rtol=1e-5, atol=1e-6):
        raise ValueError("Prepared returns and standardization do not agree.")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--news", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--price-sheet")
    parser.add_argument("--news-sheet")
    args = parser.parse_args()
    print("Reading price and news workbooks...", flush=True)
    info = prepare_data(args.prices, args.news, args.output, args.train_fraction,
                        args.val_fraction, args.price_sheet, args.news_sheet)
    print(json.dumps({"output": str(args.output), "n_stocks": info["n_stocks"],
                      "n_samples": info["n_samples"], "splits": info["splits"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
