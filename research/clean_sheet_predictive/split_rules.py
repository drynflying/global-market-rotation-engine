from __future__ import annotations

import pandas as pd

from .feature_definitions import OUTCOME_HORIZONS


def training_rows_as_of(dataset: pd.DataFrame, horizon: int, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Return only rows whose H-bar outcome was fully observable by `as_of`.

    This is the core no-look-ahead rule for R2/R3 walk-forward research. A prediction
    dated before `as_of` is still excluded when its horizon-specific outcome had not yet
    completed by the training cutoff.
    """
    if horizon not in OUTCOME_HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}; expected one of {OUTCOME_HORIZONS}")
    cutoff = pd.Timestamp(as_of)
    end_col = f"outcome_end_date_{horizon}"
    target_col = f"fwd_return_{horizon}"
    if end_col not in dataset.columns or target_col not in dataset.columns:
        raise ValueError(f"Dataset is missing {end_col} or {target_col}")
    end = pd.to_datetime(dataset[end_col], errors="coerce")
    row_date = pd.to_datetime(dataset["date"], errors="coerce")
    mask = (row_date < cutoff) & end.notna() & (end <= cutoff) & dataset[target_col].notna()
    return dataset.loc[mask].copy()


def assert_training_cutoff_integrity(rows: pd.DataFrame, horizon: int, as_of: str | pd.Timestamp) -> None:
    if rows.empty:
        return
    cutoff = pd.Timestamp(as_of)
    end = pd.to_datetime(rows[f"outcome_end_date_{horizon}"], errors="raise")
    dates = pd.to_datetime(rows["date"], errors="raise")
    if not (dates < cutoff).all():
        raise AssertionError("Training rows include observations dated on/after cutoff")
    if not (end <= cutoff).all():
        raise AssertionError("Training rows include outcomes that had not matured by cutoff")
