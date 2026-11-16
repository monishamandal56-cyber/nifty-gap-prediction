"""
src/feature_engineering.py
--------------------------
Point-in-time feature construction for Nifty 50 gap prediction.

Every feature for date t is strictly observable before 09:15 IST on date t.
Global markets (US, Europe, commodities, FX) are aligned with backward
merge_asof -- we take the most recent close BEFORE the Nifty date, never
same-day. Nikkei is also aligned backward because its same-day close
(15:00 JST = 11:30 IST) is AFTER NSE's 09:15 IST open.
"""

import pandas as pd
import numpy as np


FEATURE_COLS = [
    'sp500_ret', 'nasdaq_ret', 'dow_ret',
    'inda_ret', 'inda_intraday',
    'crude_ret', 'usdinr_ret', 'nikkei_ret',
    'nifty_prev_ret', 'nifty_prev_hl_pct', 'nifty_prev_gap',
    'nifty_prev_upper_wick', 'nifty_prev_lower_wick',
    'nifty_5d_mom', 'nifty_20d_mom',
    'day_of_week',
    'vix_level', 'vix_change',
]


def _reset_index_date(df: pd.DataFrame) -> pd.DataFrame:
    """reset_index with an explicit 'Date' column, robust to any index name."""
    out = df.reset_index()
    out.columns = ['Date'] + list(out.columns[1:])
    return out


def _ret_frame(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """Compute pct-change return frame with a named column."""
    out = pd.DataFrame(index=df.index)
    out[colname] = df['Close'].pct_change() * 100
    return out


# ---------------------------------------------------------------------------
# Historical training set
# ---------------------------------------------------------------------------

def build_features_and_targets(raw_data: dict,
                               keep_last: bool = False) -> pd.DataFrame:
    """
    Build the historical training dataset, aligned on Nifty trading dates.

    Input : raw_data  = {key: OHLCV DataFrame} from data_loader
            keep_last = if True, do not drop the final row when its target
                        (gap_pct) is NaN. Used by the live prediction path
                        where the last row is tomorrow's synthetic row.
    Output: DataFrame with columns
              [Date, prev_close, gap_pct, gap_points, <FEATURE_COLS>]
            One row per Nifty trading day where all features are available.
    """
    if 'nifty' not in raw_data or raw_data['nifty'].empty:
        raise ValueError("Nifty 50 data is required.")

    nifty = raw_data['nifty'].copy().sort_index()

    # ---- Target ------------------------------------------------------------
    nifty['prev_close'] = nifty['Close'].shift(1)
    nifty['gap_pct']    = (nifty['Open'] - nifty['prev_close']) / nifty['prev_close'] * 100
    nifty['gap_points'] = nifty['Open'] - nifty['prev_close']

    # ---- Nifty internals from t-1 -----------------------------------------
    nifty['nifty_prev_ret'] = (
        (nifty['Close'].shift(1) - nifty['Open'].shift(1))
        / nifty['Open'].shift(1) * 100
    )
    nifty['nifty_prev_hl_pct'] = (
        (nifty['High'].shift(1) - nifty['Low'].shift(1))
        / nifty['Close'].shift(1) * 100
    )
    nifty['nifty_prev_gap'] = nifty['gap_pct'].shift(1)

    body_high = nifty[['Open', 'Close']].shift(1).max(axis=1)
    body_low  = nifty[['Open', 'Close']].shift(1).min(axis=1)
    nifty['nifty_prev_upper_wick'] = (
        (nifty['High'].shift(1) - body_high) / nifty['Close'].shift(1) * 100
    )
    nifty['nifty_prev_lower_wick'] = (
        (body_low - nifty['Low'].shift(1)) / nifty['Close'].shift(1) * 100
    )

    nifty['nifty_5d_mom'] = (
        (nifty['Close'].shift(1) - nifty['Close'].shift(6))
        / nifty['Close'].shift(6) * 100
    )
    nifty['nifty_20d_mom'] = (
        (nifty['Close'].shift(1) - nifty['Close'].shift(21))
        / nifty['Close'].shift(21) * 100
    )
    nifty['day_of_week'] = nifty.index.dayofweek

    # ---- Global return frames ---------------------------------------------
    sp_df     = _ret_frame(raw_data['sp500'],  'sp500_ret')  if 'sp500'  in raw_data and not raw_data['sp500'].empty  else pd.DataFrame()
    nasdaq_df = _ret_frame(raw_data['nasdaq'], 'nasdaq_ret') if 'nasdaq' in raw_data and not raw_data['nasdaq'].empty else pd.DataFrame()
    dow_df    = _ret_frame(raw_data['dow'],    'dow_ret')    if 'dow'    in raw_data and not raw_data['dow'].empty    else pd.DataFrame()
    crude_df  = _ret_frame(raw_data['crude'],  'crude_ret')  if 'crude'  in raw_data and not raw_data['crude'].empty  else pd.DataFrame()
    usdinr_df = _ret_frame(raw_data['usdinr'], 'usdinr_ret') if 'usdinr' in raw_data and not raw_data['usdinr'].empty else pd.DataFrame()
    nikkei_df = _ret_frame(raw_data['nikkei'], 'nikkei_ret') if 'nikkei' in raw_data and not raw_data['nikkei'].empty else pd.DataFrame()

    if 'inda' in raw_data and not raw_data['inda'].empty:
        inda_df = pd.DataFrame(index=raw_data['inda'].index)
        inda_df['inda_ret'] = raw_data['inda']['Close'].pct_change() * 100
        inda_df['inda_intraday'] = (
            (raw_data['inda']['Close'] - raw_data['inda']['Open'])
            / raw_data['inda']['Open'] * 100
        )
    else:
        inda_df = pd.DataFrame()

    if 'vix' in raw_data and not raw_data['vix'].empty:
        vix_df = pd.DataFrame(index=raw_data['vix'].index)
        # India VIX closes at 15:30 IST. Like every other series it is aligned
        # with a strictly-prior backward merge (allow_exact_matches=False), so
        # Nifty date t gets the last VIX close before t. No manual shift.
        vix_df['vix_level']  = raw_data['vix']['Close']
        vix_df['vix_change'] = raw_data['vix']['Close'].pct_change() * 100
    else:
        vix_df = pd.DataFrame()

    # ---- Alignment via merge_asof -----------------------------------------
    # allow_exact_matches=False -> strictly prior date
    # allow_exact_matches=True  -> same date allowed
    base = _reset_index_date(nifty).sort_values('Date')

    merges = [
        (sp_df,      False),
        (nasdaq_df,  False),
        (dow_df,     False),
        (inda_df,    False),
        (crude_df,   False),
        (usdinr_df,  False),
        # Nikkei's same-day close (15:00 JST = 11:30 IST) is AFTER 09:15 IST,
        # so we use backward-only alignment -> picks Nikkei close on t-1.
        (nikkei_df,  False),
        (vix_df,     False),
    ]
    for df, allow_exact in merges:
        if df is None or df.empty or len(df.columns) == 0:
            continue
        clean = _reset_index_date(df.dropna()).sort_values('Date')
        
        # FIX: Force both date columns to the exact same nanosecond resolution
        base['Date'] = pd.to_datetime(base['Date']).astype('datetime64[ns]')
        clean['Date'] = pd.to_datetime(clean['Date']).astype('datetime64[ns]')
        
        base = pd.merge_asof(
            base, clean,
            on='Date',
            direction='backward',
            allow_exact_matches=allow_exact,
        )
    
    # ---- Drop incomplete rows ---------------------------------------------
    # Features must always be complete. Targets are only required for
    # training rows; when keep_last=True we let the final row survive even
    # if its target is NaN, so the live feature vector can be extracted.
    required = [c for c in FEATURE_COLS if c in base.columns]
    if not keep_last:
        required = ['gap_pct', 'gap_points', 'prev_close'] + required

    dataset = base.dropna(subset=required).reset_index(drop=True)

    return dataset


# ---------------------------------------------------------------------------
# Live feature vector for tomorrow
# ---------------------------------------------------------------------------

def prepare_live_features_for_tomorrow(raw_data: dict,
                                       scenario_overrides: dict = None):
    """
    Build the feature vector for tomorrow's opening gap prediction by
    appending a synthetic next-day Nifty row and running the SAME pipeline
    used in training. This guarantees no train/serve skew.

    Returns: (feat_df, last_nifty_close, last_nifty_date_str)
    """
    nifty = raw_data['nifty'].copy().sort_index()
    last_nifty = nifty.iloc[-1]
    last_close = float(last_nifty['Close'])
    last_date  = nifty.index[-1].strftime('%Y-%m-%d')

    # Compute the next business day (skip Sat/Sun)
    next_day = nifty.index[-1] + pd.Timedelta(days=1)
    while next_day.dayofweek >= 5:
        next_day += pd.Timedelta(days=1)

    # Append a synthetic next-day row with NaN OHLCV. Its target will be NaN;
    # keep_last=True ensures it is not dropped by the completeness filter.
    synthetic = pd.DataFrame(
        {c: [np.nan] for c in ['Open', 'High', 'Low', 'Close', 'Volume']},
        index=[next_day],
    )
    nifty_ext = pd.concat([nifty, synthetic])

    raw_ext = dict(raw_data)
    raw_ext['nifty'] = nifty_ext

    feat_full = build_features_and_targets(raw_ext, keep_last=True)

    if len(feat_full) == 0:
        raise RuntimeError(
            "prepare_live_features_for_tomorrow: no feature rows produced. "
            "Check that raw_data contains fresh Nifty and global tickers."
        )

    last_row = feat_full.iloc[-1]
    if pd.Timestamp(last_row['Date']).normalize() != pd.Timestamp(next_day).normalize():
        raise RuntimeError(
            "Live feature row for %s was dropped (an input is missing or the "
            "history window is too short). Refusing to predict from an older "
            "row. Last available row: %s." % (next_day.date(), last_row['Date'])
        )
    features = {c: float(last_row[c]) for c in FEATURE_COLS}

    # Apply scenario overrides (used for the What-If matrix)
    if scenario_overrides:
        for k, v in scenario_overrides.items():
            if k in features and v is not None:
                features[k] = float(v)

    feat_df = pd.DataFrame([features])[FEATURE_COLS]
    return feat_df, last_close, last_date
