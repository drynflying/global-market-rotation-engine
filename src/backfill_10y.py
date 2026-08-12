from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.common import ROOT, load_config


OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
OUTPUT_DIR = ROOT / "backtest_data"
OUTPUT_OHLCV = OUTPUT_DIR / "ohlcv_10y.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "backfill_summary.json"
OUTPUT_FAILURES = OUTPUT_DIR / "fetch_failures.csv"
OUTPUT_CONFIG = OUTPUT_DIR / "rotation_universe_snapshot.csv"


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {value}")
    return value


def _normalize_download(raw: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    found: set[str] = set()

    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS), symbols

    for symbol in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(map(str, raw.columns.get_level_values(0)))
                level1 = set(map(str, raw.columns.get_level_values(1)))
                if symbol in level0:
                    one = raw[symbol].copy()
                elif symbol in level1:
                    one = raw.xs(symbol, axis=1, level=1).copy()
                else:
                    continue
            else:
                if len(symbols) != 1:
                    continue
                one = raw.copy()

            required = ["Open", "High", "Low", "Close", "Volume"]
            if not set(required).issubset(one.columns):
                continue

            one = one[required].reset_index()
            date_col = "Date" if "Date" in one.columns else one.columns[0]
            one = one.rename(
                columns={
                    date_col: "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            one["date"] = pd.to_datetime(one["date"], errors="coerce")
            try:
                one["date"] = one["date"].dt.tz_localize(None)
            except TypeError:
                pass

            for col in ["open", "high", "low", "close", "volume"]:
                one[col] = pd.to_numeric(one[col], errors="coerce")

            one["ticker"] = symbol
            one = one.dropna(subset=["date", "close"])
            one = one[one["close"] > 0]
            one = one[one["volume"].fillna(0) >= 0]

            if not one.empty:
                frames.append(one[OHLCV_COLUMNS])
                found.add(symbol)
        except Exception as exc:  # noqa: BLE001 - keep the backfill going for other symbols
            print(f"WARNING: Could not normalize {symbol}: {exc}")

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=OHLCV_COLUMNS)
    )
    missing = [symbol for symbol in symbols if symbol not in found]
    return combined, missing


def _download(
    symbols: list[str],
    start_date: str,
    end_date_exclusive: str,
) -> tuple[pd.DataFrame, list[str]]:
    # Keep this aligned with the production fetcher: auto_adjust=True provides
    # split/dividend-adjusted OHLC where Yahoo/yfinance supplies adjustments.
    raw = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date_exclusive,
        interval="1d",
        auto_adjust=True,
        actions=False,
        repair=True,
        progress=False,
        threads=True,
        group_by="ticker",
        timeout=60,
        multi_level_index=True,
    )
    return _normalize_download(raw, symbols)


def _build_summary(
    data: pd.DataFrame,
    symbols: list[str],
    failures: pd.DataFrame,
    start_date: str,
    end_date_exclusive: str,
    backtest_years: int,
    warmup_years: int,
) -> dict:
    per_ticker: list[dict] = []

    for ticker in symbols:
        g = data[data["ticker"] == ticker].sort_values("date")
        if g.empty:
            continue
        earliest = pd.Timestamp(g["date"].iloc[0])
        latest = pd.Timestamp(g["date"].iloc[-1])
        per_ticker.append(
            {
                "ticker": ticker,
                "bars": int(len(g)),
                "earliest_date": earliest.strftime("%Y-%m-%d"),
                "latest_date": latest.strftime("%Y-%m-%d"),
                "calendar_years_covered": round((latest - earliest).days / 365.25, 2),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "adjustment": "auto_adjust=True; actions=False; repair=True",
        "requested_backtest_years": backtest_years,
        "requested_warmup_years": warmup_years,
        "requested_total_calendar_years": backtest_years + warmup_years,
        "requested_start_date": start_date,
        "requested_end_date_exclusive": end_date_exclusive,
        "configured_symbols": len(symbols),
        "symbols_with_data": int(data["ticker"].nunique()),
        "failed_symbols": int(len(failures)),
        "total_rows": int(len(data)),
        "overall_earliest_date": (
            pd.Timestamp(data["date"].min()).strftime("%Y-%m-%d") if not data.empty else None
        ),
        "overall_latest_date": (
            pd.Timestamp(data["date"].max()).strftime("%Y-%m-%d") if not data.empty else None
        ),
        "duplicate_ticker_date_rows": int(data.duplicated(["ticker", "date"]).sum()),
        "per_ticker": per_ticker,
    }


def main() -> None:
    backtest_years = _env_int("BACKFILL_YEARS", 10, minimum=1)
    warmup_years = _env_int("BACKFILL_WARMUP_YEARS", 1, minimum=0)
    batch_size = _env_int("BACKFILL_BATCH_SIZE", 25, minimum=1)

    # Yahoo's `end` argument is exclusive. Request through tomorrow so today's
    # completed observation is eligible if the feed has finalized it.
    today_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start = today_utc - pd.DateOffset(years=backtest_years + warmup_years)
    end_exclusive = today_utc + pd.Timedelta(days=1)
    start_date = start.strftime("%Y-%m-%d")
    end_date_exclusive = end_exclusive.strftime("%Y-%m-%d")

    cfg = load_config()
    active = cfg[cfg["enabled"] & cfg["query_ohlcv"]].copy()
    symbols = sorted(
        active["query_symbol"]
        .replace("", np.nan)
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    if not symbols:
        raise RuntimeError("No enabled/query_ohlcv symbols were found in rotation_universe.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    active.to_csv(OUTPUT_CONFIG, index=False)

    print(
        f"Backfilling {len(symbols)} symbols from {start_date} through "
        f"{end_date_exclusive} (end exclusive)."
    )
    print(
        f"Target: {backtest_years} backtest years + {warmup_years} warm-up year(s)."
    )

    frames: list[pd.DataFrame] = []
    retry_symbols: list[str] = []
    failures: list[dict] = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        print(f"Batch {i // batch_size + 1}: {len(batch)} symbols")
        try:
            frame, missing = _download(batch, start_date, end_date_exclusive)
            if not frame.empty:
                frames.append(frame)
            retry_symbols.extend(missing)
        except Exception as exc:  # noqa: BLE001 - retry individually below
            print(f"WARNING: Batch failed: {exc}")
            retry_symbols.extend(batch)
        time.sleep(1)

    for symbol in sorted(set(retry_symbols)):
        print(f"Retrying {symbol} individually.")
        try:
            frame, missing = _download([symbol], start_date, end_date_exclusive)
            if not frame.empty:
                frames.append(frame)
            if missing:
                failures.append({"ticker": symbol, "reason": "No usable OHLCV returned"})
        except Exception as exc:  # noqa: BLE001 - record and continue
            failures.append({"ticker": symbol, "reason": str(exc)})
        time.sleep(0.25)

    if not frames:
        raise RuntimeError("No market data was returned for any configured symbol.")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "ticker", "close"])
    data = (
        data.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )

    # Enforce the requested window even if an upstream response includes an
    # unexpected extra observation.
    data = data[(data["date"] >= start) & (data["date"] < end_exclusive)].copy()

    failure_df = pd.DataFrame(failures, columns=["ticker", "reason"])
    summary = _build_summary(
        data=data,
        symbols=symbols,
        failures=failure_df,
        start_date=start_date,
        end_date_exclusive=end_date_exclusive,
        backtest_years=backtest_years,
        warmup_years=warmup_years,
    )

    data.to_csv(
        OUTPUT_OHLCV,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.6f",
    )
    failure_df.to_csv(OUTPUT_FAILURES, index=False)
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBackfill complete")
    print(f"Rows: {summary['total_rows']:,}")
    print(f"Symbols with data: {summary['symbols_with_data']} / {summary['configured_symbols']}")
    print(f"Failures: {summary['failed_symbols']}")
    print(f"Range: {summary['overall_earliest_date']} -> {summary['overall_latest_date']}")
    print(f"Saved: {OUTPUT_OHLCV.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_SUMMARY.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_FAILURES.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_CONFIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
