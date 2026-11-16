"""
src/predictor.py
----------------
End-to-end orchestration for Nifty 50 next-day gap prediction.

Flow:
    load_all_historical_data  ->  build_features_and_targets
        ->  train_and_save_model  (only if missing or forced)
    load_all_historical_data (fresh)  ->  prepare_live_features_for_tomorrow
        ->  model.predict_details
    plus a 9-point What-If matrix for overnight S&P 500 swings,
    plus a live ticker snapshot for the dashboard.
"""

import pandas as pd

from src.data_loader import (
    load_all_historical_data,
    get_latest_market_snapshot,
)
from src.feature_engineering import (
    build_features_and_targets,
    prepare_live_features_for_tomorrow,
)
from src.models import (
    train_and_save_model,
    load_trained_model,
    GapPredictionEnsemble,
)


# Human-readable labels for XAI drivers
DRIVER_LABELS = {
    'sp500_ret':              'S&P 500 Return',
    'nasdaq_ret':             'Nasdaq Return',
    'dow_ret':                'Dow Jones Return',
    'inda_ret':               'US-Traded India ETF (INDA)',
    'inda_intraday':          'INDA Intraday Trend',
    'crude_ret':              'Crude Oil Move',
    'usdinr_ret':             'USD/INR Currency Move',
    'nikkei_ret':             'Nikkei 225 Morning Trend',
    'nifty_prev_ret':         'Nifty Intraday Return',
    'nifty_prev_hl_pct':      'Nifty Daily Volatility Range',
    'nifty_prev_gap':         'Prior Day Opening Gap',
    'nifty_prev_upper_wick':  'Upper Wick (Selling Rejection)',
    'nifty_prev_lower_wick':  'Lower Wick (Dip Buying Support)',
    'nifty_5d_mom':           'Nifty 5-Day Trend',
    'nifty_20d_mom':          'Nifty 20-Day Trend',
    'day_of_week':            'Day of Week Seasonality',
    'vix_level':              'India VIX Level',
    'vix_change':             'India VIX Daily Shift',
}


# ---------------------------------------------------------------------------
# Model loading / training
# ---------------------------------------------------------------------------

def get_or_train_model(force_retrain: bool = False,
                       weights: dict = None,
                       model_set: str = "paper"):
    """
    Load the trained ensemble from disk, or train a fresh one if missing
    or if force_retrain is True.

    weights / model_set let you swap the production configuration without
    editing code -- e.g. equal weights, or the XGBoost model set.
    """
    model = load_trained_model()
    if model is None or force_retrain:
        print("Training Nifty 50 gap prediction model on historical data...")
        raw     = load_all_historical_data(period="5y", force_refresh=force_retrain)
        dataset = build_features_and_targets(raw)
        model, _ = train_and_save_model(
            dataset,
            weights=weights,
            model_set=model_set,
        )
    return model


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_tomorrow(scenario_overrides: dict = None,
                     force_retrain: bool = False,
                     include_snapshot: bool = True) -> dict:
    """
    Predict tomorrow's Nifty 50 opening gap.

    Parameters
    ----------
    scenario_overrides : optional dict of feature overrides for What-If runs
    force_retrain      : retrain the model before predicting
    include_snapshot   : also fetch a live ticker snapshot (slower)

    Returns
    -------
    dict with keys:
        gap_pct, gap_points, last_close, predicted_open,
        expected_low, expected_high, direction,
        prob_up, prob_flat, prob_down,
        drivers_formatted, confidence_band_pct,
        date_of_last_close, validation_metrics,
        scenario_matrix, live_snapshot
    """
    model = get_or_train_model(force_retrain=force_retrain)

    # Fresh 3-month window (20-day momentum needs >=21 prior sessions) for the live feature vector
    raw_live = load_all_historical_data(period="3mo")
    feat_df, last_close, last_date = prepare_live_features_for_tomorrow(
        raw_live, scenario_overrides
    )

    # Core prediction
    result = model.predict_details(feat_df, last_close)
    result['date_of_last_close'] = last_date
    result['validation_metrics'] = getattr(model, 'metrics', {})

    # Human-readable XAI drivers
    friendly_drivers = []
    for feat_name, info in result['drivers']:
        friendly_drivers.append({
            'code': feat_name,
            'name': DRIVER_LABELS.get(feat_name, feat_name),
            'current_value': info['value'],
            'impact_pct': info['contribution_pct'],
        })
    result['drivers_formatted'] = friendly_drivers

    # ---- What-If scenario matrix --------------------------------------
    # Simulate 9 overnight S&P 500 moves and their knock-on effects on
    # Nasdaq and INDA, using the historical beta ratios observed in the
    # training data. If the model was not trained on historical data in
    # this session, we fall back to fixed ratios.
    sp_vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    nasdaq_mult = 1.20
    inda_mult   = 0.80

    matrix = []
    for sp_val in sp_vals:
        overrides = {
            'sp500_ret':  sp_val,
            'nasdaq_ret': sp_val * nasdaq_mult,
            'inda_ret':   sp_val * inda_mult,
        }
        temp_feat, _, _ = prepare_live_features_for_tomorrow(raw_live, overrides)
        sim = model.predict_details(temp_feat, last_close)
        matrix.append({
            'sp500_scenario_pct':   sp_val,
            'predicted_gap_pct':    sim['gap_pct'],
            'predicted_gap_points': sim['gap_points'],
            'predicted_open':       sim['predicted_open'],
            'direction':            sim['direction'],
            'prob_up':              sim['prob_up'],
            'prob_flat':            sim['prob_flat'],
            'prob_down':            sim['prob_down'],
        })
    result['scenario_matrix'] = matrix

    # ---- Optional live snapshot ---------------------------------------
    if include_snapshot:
        result['live_snapshot'] = get_latest_market_snapshot()
    else:
        result['live_snapshot'] = {}

    return result


# ---------------------------------------------------------------------------
# Pretty print helper (used by CLI and __main__)
# ---------------------------------------------------------------------------

def print_prediction(res: dict) -> None:
    print("\n--- Nifty 50 Next-Day Gap Prediction ---")
    print(f"Last Nifty Close : {res['last_close']:.2f}  ({res['date_of_last_close']})")
    print(f"Predicted Gap    : {res['gap_pct']:+.2f}%  ({res['gap_points']:+.1f} pts)")
    print(f"Predicted Open   : {res['predicted_open']:.2f}")
    print(f"80% Range        : {res['expected_low']:.2f}  to  {res['expected_high']:.2f}")
    print(f"Direction        : {res['direction']}")
    print(f"Probabilities    : Up {res['prob_up']}%   "
          f"Flat {res['prob_flat']}%   Down {res['prob_down']}%")
    print("\nTop drivers:")
    for d in res.get('drivers_formatted', [])[:6]:
        sign = "+" if d['impact_pct'] >= 0 else "-"
        print(f"  {d['name']:<32s} value={d['current_value']:>7.2f}   "
              f"impact={sign}{abs(d['impact_pct']):.3f}%")

    if res.get('validation_metrics'):
        m = res['validation_metrics']
        print("\nValidation metrics (walk-forward OOF):")
        for k, v in m.items():
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    res = predict_tomorrow()
    print_prediction(res)