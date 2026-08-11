from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from src.common import OHLCV_PATH, as_bool


OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


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

            one["ticker"] = symbol
            one = one.dropna(subset=["date", "close"])
            if not one.empty:
                frames.append(one[OHLCV_COLUMNS])
                found.add(symbol)
        except Exception as exc:
            print(f"WARNING: Could not normalize {symbol}: {exc}")

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=OHLCV_COLUMNS)
    )
    missing = [s for s in symbols if s not in found]
    return combined, missing


def _download(symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    # auto_adjust=True means the OHLC series is adjusted for splits/dividends
    # where yfinance provides the adjustment. That gives cleaner return history.
    raw = yf.download(
        tickers=symbols,
        period="2y",
        interval="1d",
        auto_adjust=True,
        actions=False,
        repair=True,
        progress=False,
        threads=True,
        group_by="ticker",
        timeout=30,
        multi_level_index=True,
    )
    return _normalize_download(raw, symbols)


def fetch_market_data(cfg: pd.DataFrame, batch_size: int = 35) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = sorted(
        cfg.loc[cfg["enabled"] & cfg["query_ohlcv"], "query_symbol"]
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    frames: list[pd.DataFrame] = []
    retry_symbols: list[str] = []
    failures: list[dict] = []

    print(f"Fetching approximately 2 years of daily OHLCV for {len(symbols)} symbols.")

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"Batch {i // batch_size + 1}: {len(batch)} symbols")
        try:
            frame, missing = _download(batch)
            if not frame.empty:
                frames.append(frame)
            retry_symbols.extend(missing)
        except Exception as exc:
            print(f"WARNING: Batch failed: {exc}")
            retry_symbols.extend(batch)
        time.sleep(1)

    for symbol in sorted(set(retry_symbols)):
        print(f"Retrying {symbol} individually.")
        try:
            frame, missing = _download([symbol])
            if not frame.empty:
                frames.append(frame)
            if missing:
                failures.append({"ticker": symbol, "reason": "No usable OHLCV returned"})
        except Exception as exc:
            failures.append({"ticker": symbol, "reason": str(exc)})

    if not frames:
        raise RuntimeError("No market data was returned for any configured symbol.")

    fetched = pd.concat(frames, ignore_index=True)
    fetched["date"] = pd.to_datetime(fetched["date"])
    fetched = (
        fetched.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )

    return fetched, pd.DataFrame(failures, columns=["ticker", "reason"])


def update_ohlcv_history(fetched: pd.DataFrame) -> pd.DataFrame:
    if OHLCV_PATH.exists():
        old = pd.read_csv(OHLCV_PATH, parse_dates=["date"])
        combined = pd.concat([old, fetched], ignore_index=True)
    else:
        combined = fetched.copy()

    combined["date"] = pd.to_datetime(combined["date"])
    combined = (
        combined.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )

    OHLCV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        OHLCV_PATH,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.6f",
    )
    return combined
