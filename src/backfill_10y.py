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
OUTPUT_ATTEMPTS = OUTPUT_DIR / "fetch_attempts.csv"
OUTPUT_QUALITY = OUTPUT_DIR / "ticker_quality.csv"
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


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric; received {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {value}")
    return value


def _env_symbol_set(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {
        value.strip().upper()
        for value in raw.split(",")
        if value.strip()
    }


def _normalize_download(raw: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    found: set[str] = set()

    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS), list(symbols)

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
        except Exception as exc:  # noqa: BLE001
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
    *,
    threads: bool,
    timeout: int = 60,
) -> tuple[pd.DataFrame, list[str]]:
    raw = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date_exclusive,
        interval="1d",
        auto_adjust=True,
        actions=False,
        repair=True,
        progress=False,
        threads=threads,
        group_by="ticker",
        timeout=timeout,
        multi_level_index=True,
    )
    return _normalize_download(raw, symbols)


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    data = pd.concat(nonempty, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data = data.dropna(subset=["date", "ticker", "close"])
    return (
        data.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )


def _symbols_present(data: pd.DataFrame) -> set[str]:
    if data.empty:
        return set()
    return set(data["ticker"].dropna().astype(str).str.upper().str.strip())


def _record_attempt(
    attempts: list[dict],
    *,
    stage: str,
    symbols: list[str],
    frame: pd.DataFrame | None,
    missing: list[str],
    error: str | None,
) -> None:
    frame = frame if frame is not None else pd.DataFrame(columns=OHLCV_COLUMNS)
    found = _symbols_present(frame)
    for symbol in symbols:
        g = frame[frame["ticker"].eq(symbol)] if not frame.empty else frame
        attempts.append(
            {
                "stage": stage,
                "ticker": symbol,
                "success": symbol in found and symbol not in set(missing),
                "rows_returned": int(len(g)),
                "earliest_date": (
                    pd.Timestamp(g["date"].min()).strftime("%Y-%m-%d")
                    if not g.empty else None
                ),
                "latest_date": (
                    pd.Timestamp(g["date"].max()).strftime("%Y-%m-%d")
                    if not g.empty else None
                ),
                "error": error,
            }
        )


def _download_segmented(
    symbol: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    *,
    segment_years: int,
    pause_seconds: float,
    attempts: list[dict],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    segment_start = pd.Timestamp(start)

    while segment_start < end_exclusive:
        segment_end = min(
            segment_start + pd.DateOffset(years=segment_years),
            end_exclusive,
        )
        stage = (
            f"segmented_{segment_start.strftime('%Y%m%d')}_"
            f"{segment_end.strftime('%Y%m%d')}"
        )
        try:
            frame, missing = _download(
                [symbol],
                segment_start.strftime("%Y-%m-%d"),
                segment_end.strftime("%Y-%m-%d"),
                threads=False,
            )
            _record_attempt(
                attempts,
                stage=stage,
                symbols=[symbol],
                frame=frame,
                missing=missing,
                error=None,
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            _record_attempt(
                attempts,
                stage=stage,
                symbols=[symbol],
                frame=None,
                missing=[symbol],
                error=str(exc),
            )
        segment_start = segment_end
        if pause_seconds:
            time.sleep(pause_seconds)

    return _merge_frames(frames)


def _assess_symbol_quality(
    data: pd.DataFrame,
    symbols: list[str],
    *,
    global_latest: pd.Timestamp | None,
    min_bars: int,
    min_coverage_ratio: float,
    max_gap_days: int,
    max_stale_days: int,
    allowed_missing: set[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol in symbols:
        g = data[data["ticker"].eq(symbol)].sort_values("date")
        reasons: list[str] = []

        if g.empty:
            reasons.append("no usable OHLCV")
            bars = 0
            earliest = None
            latest = None
            coverage_ratio = 0.0
            max_gap = None
            stale_days = None
        else:
            bars = int(len(g))
            earliest_ts = pd.Timestamp(g["date"].iloc[0])
            latest_ts = pd.Timestamp(g["date"].iloc[-1])
            earliest = earliest_ts.strftime("%Y-%m-%d")
            latest = latest_ts.strftime("%Y-%m-%d")

            business_days = len(pd.bdate_range(earliest_ts.normalize(), latest_ts.normalize()))
            coverage_ratio = bars / max(business_days, 1)

            gaps = g["date"].diff().dt.days.dropna()
            max_gap = int(gaps.max()) if not gaps.empty else 0

            stale_days = (
                int((global_latest.normalize() - latest_ts.normalize()).days)
                if global_latest is not None else 0
            )

            if bars < min_bars:
                reasons.append(f"only {bars} bars (< {min_bars})")
            if coverage_ratio < min_coverage_ratio:
                reasons.append(
                    f"business-day coverage {coverage_ratio:.3f} (< {min_coverage_ratio:.3f})"
                )
            if max_gap > max_gap_days:
                reasons.append(f"max internal calendar gap {max_gap}d (> {max_gap_days}d)")
            if stale_days > max_stale_days:
                reasons.append(f"latest observation stale by {stale_days}d (> {max_stale_days}d)")

        allowlisted = symbol in allowed_missing
        quality_ok = not reasons or allowlisted
        rows.append(
            {
                "ticker": symbol,
                "quality_ok": quality_ok,
                "allowlisted": allowlisted,
                "bars": bars,
                "earliest_date": earliest,
                "latest_date": latest,
                "business_day_coverage_ratio": round(float(coverage_ratio), 6),
                "max_internal_gap_days": max_gap,
                "stale_days_vs_dataset_latest": stale_days,
                "reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def _build_summary(
    data: pd.DataFrame,
    symbols: list[str],
    quality: pd.DataFrame,
    attempts: pd.DataFrame,
    start_date: str,
    end_date_exclusive: str,
    backtest_years: int,
    warmup_years: int,
    allowed_missing: set[str],
) -> dict:
    failed = quality[~quality["quality_ok"]]
    raw_missing = quality[quality["bars"].eq(0)]
    recovered = []
    if not attempts.empty:
        first_success_stage = (
            attempts[attempts["success"]]
            .drop_duplicates("ticker", keep="first")
            .set_index("ticker")["stage"]
            .to_dict()
        )
        recovered = sorted(
            ticker
            for ticker, stage in first_success_stage.items()
            if stage != "bulk_pass_1"
        )

    per_ticker = quality.to_dict(orient="records")
    overall_earliest = (
        pd.Timestamp(data["date"].min()).strftime("%Y-%m-%d")
        if not data.empty else None
    )
    overall_latest = (
        pd.Timestamp(data["date"].max()).strftime("%Y-%m-%d")
        if not data.empty else None
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
        "symbols_missing_raw": int(len(raw_missing)),
        "failed_symbols": int(len(failed)),
        "failed_symbol_list": failed["ticker"].tolist(),
        "allowed_missing_symbols": sorted(allowed_missing),
        "universe_completeness_status": "ok" if failed.empty else "failed",
        "completeness_ratio": round(
            float((len(symbols) - len(failed)) / max(len(symbols), 1)),
            6,
        ),
        "symbols_recovered_after_first_bulk_pass": recovered,
        "recovered_after_first_bulk_pass_count": len(recovered),
        "total_rows": int(len(data)),
        "overall_earliest_date": overall_earliest,
        "overall_latest_date": overall_latest,
        "duplicate_ticker_date_rows": int(data.duplicated(["ticker", "date"]).sum()),
        "fetch_attempt_rows": int(len(attempts)),
        "per_ticker": per_ticker,
    }


def main() -> None:
    backtest_years = _env_int("BACKFILL_YEARS", 10, minimum=1)
    warmup_years = _env_int("BACKFILL_WARMUP_YEARS", 1, minimum=0)
    batch_size = _env_int("BACKFILL_BATCH_SIZE", 10, minimum=1)
    second_pass_batch_size = _env_int("BACKFILL_SECOND_PASS_BATCH_SIZE", 5, minimum=1)
    individual_attempts = _env_int("BACKFILL_INDIVIDUAL_ATTEMPTS", 3, minimum=1)
    segment_years = _env_int("BACKFILL_SEGMENT_YEARS", 3, minimum=1)
    min_bars = _env_int("BACKFILL_MIN_BARS", 260, minimum=1)
    max_gap_days = _env_int("BACKFILL_MAX_GAP_DAYS", 21, minimum=1)
    max_stale_days = _env_int("BACKFILL_MAX_STALE_DAYS", 7, minimum=0)
    retry_base_seconds = _env_float("BACKFILL_RETRY_BASE_SECONDS", 1.0, minimum=0.0)
    between_stage_pause = _env_float("BACKFILL_STAGE_PAUSE_SECONDS", 5.0, minimum=0.0)
    min_coverage_ratio = _env_float("BACKFILL_MIN_COVERAGE_RATIO", 0.85, minimum=0.0)
    allowed_missing = _env_symbol_set("BACKFILL_ALLOWED_MISSING")

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

    unknown_allow = sorted(allowed_missing - set(symbols))
    if unknown_allow:
        raise ValueError(
            "BACKFILL_ALLOWED_MISSING contains symbols not in the active universe: "
            + ", ".join(unknown_allow)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    active.to_csv(OUTPUT_CONFIG, index=False)

    print(
        f"R1.1 reliable backfill: {len(symbols)} symbols from {start_date} through "
        f"{end_date_exclusive} (end exclusive)."
    )
    print(
        f"Target: {backtest_years} research years + {warmup_years} warm-up year(s). "
        f"No silent ticker omissions are permitted."
    )

    frames: list[pd.DataFrame] = []
    attempts: list[dict] = []

    # PASS 1: small batches, threads disabled for lower request concurrency.
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            frame, missing = _download(
                batch,
                start_date,
                end_date_exclusive,
                threads=False,
            )
            _record_attempt(
                attempts,
                stage="bulk_pass_1",
                symbols=batch,
                frame=frame,
                missing=missing,
                error=None,
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: bulk pass 1 failed for {batch}: {exc}")
            _record_attempt(
                attempts,
                stage="bulk_pass_1",
                symbols=batch,
                frame=None,
                missing=batch,
                error=str(exc),
            )
        if retry_base_seconds:
            time.sleep(retry_base_seconds)

    data = _merge_frames(frames)
    missing = sorted(set(symbols) - _symbols_present(data))
    print(f"After bulk pass 1: {len(symbols) - len(missing)} / {len(symbols)} present.")

    # PASS 2: retry only missing names, in even smaller batches.
    if missing and between_stage_pause:
        time.sleep(between_stage_pause)
    for i in range(0, len(missing), second_pass_batch_size):
        batch = missing[i : i + second_pass_batch_size]
        try:
            frame, still_missing = _download(
                batch,
                start_date,
                end_date_exclusive,
                threads=False,
            )
            _record_attempt(
                attempts,
                stage="bulk_pass_2",
                symbols=batch,
                frame=frame,
                missing=still_missing,
                error=None,
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: bulk pass 2 failed for {batch}: {exc}")
            _record_attempt(
                attempts,
                stage="bulk_pass_2",
                symbols=batch,
                frame=None,
                missing=batch,
                error=str(exc),
            )
        if retry_base_seconds:
            time.sleep(retry_base_seconds)

    data = _merge_frames(frames)
    missing = sorted(set(symbols) - _symbols_present(data))
    print(f"After bulk pass 2: {len(symbols) - len(missing)} / {len(symbols)} present.")

    # PASS 3+: repeated full-window individual requests with increasing delay.
    for attempt_number in range(1, individual_attempts + 1):
        if not missing:
            break
        if between_stage_pause:
            time.sleep(between_stage_pause)
        current = list(missing)
        for symbol in current:
            stage = f"individual_pass_{attempt_number}"
            try:
                frame, still_missing = _download(
                    [symbol],
                    start_date,
                    end_date_exclusive,
                    threads=False,
                )
                _record_attempt(
                    attempts,
                    stage=stage,
                    symbols=[symbol],
                    frame=frame,
                    missing=still_missing,
                    error=None,
                )
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:  # noqa: BLE001
                _record_attempt(
                    attempts,
                    stage=stage,
                    symbols=[symbol],
                    frame=None,
                    missing=[symbol],
                    error=str(exc),
                )
            if retry_base_seconds:
                time.sleep(retry_base_seconds * attempt_number)

        data = _merge_frames(frames)
        missing = sorted(set(symbols) - _symbols_present(data))
        print(
            f"After individual pass {attempt_number}: "
            f"{len(symbols) - len(missing)} / {len(symbols)} present."
        )

    # FINAL RECOVERY: split the long request into shorter segments for the
    # remaining names. Successful segments are merged; quality checks below
    # reject partial histories with material internal gaps.
    if missing:
        if between_stage_pause:
            time.sleep(between_stage_pause)
        for symbol in list(missing):
            print(f"Segmented recovery for {symbol}.")
            frame = _download_segmented(
                symbol,
                start,
                end_exclusive,
                segment_years=segment_years,
                pause_seconds=retry_base_seconds,
                attempts=attempts,
            )
            if not frame.empty:
                frames.append(frame)

    data = _merge_frames(frames)
    data = data[(data["date"] >= start) & (data["date"] < end_exclusive)].copy()

    if data.empty:
        raise RuntimeError("No market data was returned for any configured symbol.")

    global_latest = pd.Timestamp(data["date"].max())
    quality = _assess_symbol_quality(
        data,
        symbols,
        global_latest=global_latest,
        min_bars=min_bars,
        min_coverage_ratio=min_coverage_ratio,
        max_gap_days=max_gap_days,
        max_stale_days=max_stale_days,
        allowed_missing=allowed_missing,
    )
    attempt_df = pd.DataFrame(attempts)

    failures = quality[~quality["quality_ok"]][["ticker", "reason"]].copy()
    summary = _build_summary(
        data=data,
        symbols=symbols,
        quality=quality,
        attempts=attempt_df,
        start_date=start_date,
        end_date_exclusive=end_date_exclusive,
        backtest_years=backtest_years,
        warmup_years=warmup_years,
        allowed_missing=allowed_missing,
    )

    # Always write diagnostics before enforcing the research completeness gate.
    data.to_csv(
        OUTPUT_OHLCV,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.6f",
    )
    failures.to_csv(OUTPUT_FAILURES, index=False)
    attempt_df.to_csv(OUTPUT_ATTEMPTS, index=False)
    quality.to_csv(OUTPUT_QUALITY, index=False)
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nR1.1 backfill finished")
    print(f"Rows: {summary['total_rows']:,}")
    print(f"Symbols with data: {summary['symbols_with_data']} / {summary['configured_symbols']}")
    print(f"Universe completeness: {summary['universe_completeness_status']}")
    print(f"Quality failures: {summary['failed_symbols']}")
    print(
        f"Recovered after first bulk pass: "
        f"{summary['recovered_after_first_bulk_pass_count']}"
    )
    print(f"Range: {summary['overall_earliest_date']} -> {summary['overall_latest_date']}")
    print(f"Saved: {OUTPUT_OHLCV.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_SUMMARY.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_FAILURES.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_ATTEMPTS.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_QUALITY.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_CONFIG.relative_to(ROOT)}")

    unresolved = quality[~quality["quality_ok"]]
    if not unresolved.empty:
        details = "; ".join(
            f"{row.ticker}: {row.reason}"
            for row in unresolved.itertuples(index=False)
        )
        raise RuntimeError(
            "R1.1 universe completeness gate failed. "
            "R2 discovery is blocked until every active query_ohlcv symbol is usable "
            "or explicitly allowlisted via BACKFILL_ALLOWED_MISSING. "
            f"Failures: {details}"
        )


if __name__ == "__main__":
    main()
