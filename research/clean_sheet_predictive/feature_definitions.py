from __future__ import annotations

FEATURE_VERSION = "CLEAN_SHEET_R1_V1"

RETURN_WINDOWS = (21, 63, 126, 189, 252)
SMA_WINDOWS = (20, 50, 100, 200)
VOL_WINDOWS = (21, 63, 126)
RANGE_WINDOWS = (63, 126, 252)
OUTCOME_HORIZONS = (21, 63, 126, 189)

BASE_OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]

# Features are intentionally based only on OHLCV/config metadata.  No Patch 6 score,
# rotation state, AI output, Path Risk, or weekly recommendation fields are consumed.
FEATURE_FAMILIES = {
    "momentum": [
        *(f"ret_{w}" for w in RETURN_WINDOWS),
        *(f"rs_primary_{w}" for w in RETURN_WINDOWS),
        *(f"rs_spy_{w}" for w in RETURN_WINDOWS),
        "cs_ret_63_pct",
        "cs_ret_126_pct",
        "cs_rs_spy_63_pct",
        "cs_rs_spy_126_pct",
    ],
    "trend": [
        *(f"above_sma_{w}" for w in SMA_WINDOWS),
        *(f"dist_sma_{w}" for w in SMA_WINDOWS),
        *(f"sma_{w}_slope_20" for w in SMA_WINDOWS),
        "sma_50_over_200",
    ],
    "price_location": [
        *(f"dist_high_{w}" for w in RANGE_WINDOWS),
        *(f"dist_low_{w}" for w in RANGE_WINDOWS),
        *(f"range_position_{w}" for w in RANGE_WINDOWS),
        "bars_since_high_252",
        "bars_since_low_252",
    ],
    "risk": [
        *(f"vol_{w}" for w in VOL_WINDOWS),
        *(f"downside_vol_{w}" for w in VOL_WINDOWS),
        "vol_accel_21_126",
        "atr20_pct",
    ],
    "participation": [
        "cmf20",
        "cmf63",
        "dollar_volume_ratio_20_63",
        "volume_z20",
        "signed_volume_balance_20",
        "signed_volume_balance_63",
        "obv_change_20_norm",
        "obv_change_63_norm",
    ],
    "market_regime": [
        "mkt_spy_ret_21",
        "mkt_spy_ret_63",
        "mkt_spy_ret_126",
        "mkt_spy_ret_252",
        "mkt_spy_above_sma_200",
        "mkt_spy_sma_200_slope_20",
        "mkt_spy_vol_21",
        "mkt_spy_vol_accel_21_126",
        "mkt_spy_dist_high_252",
        "mkt_breadth_above_sma200",
        "mkt_dispersion_ret63",
    ],
}

FEATURE_COLUMNS = [item for items in FEATURE_FAMILIES.values() for item in items]
