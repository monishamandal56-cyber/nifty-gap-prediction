"""
src/models.py
-------------
Dual-head ML ensemble for Nifty 50 gap prediction.

CI calibration note
-------------------
The 80% prediction interval uses empirical 10th/90th percentiles of
walk-forward OOF residuals, then SCALES those percentiles by the ratio
of recent realized residual volatility to training-period residual
volatility. This adapts the band width to regime shifts: when the model
starts making larger errors than during training, the band widens
automatically. Bounded to [0.5x, 3.0x] to avoid extreme values from a
single outlier day.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_engineering import FEATURE_COLS


MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

FLAT_BAND = 0.10          # |gap%| < 0.10 -> Flat
MIN_RESID_SAMPLES = 10    # need at least this many residuals for scaling
SCALE_MIN, SCALE_MAX = 0.5, 3.0


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


class GapPredictionEnsemble:
    """Dual-head ensemble. Weights configurable for ablation variants."""

    def __init__(self,
                 weights: dict = None,
                 huber_epsilon: float = 1.35,
                 model_set: str = "paper"):
        self.model_set = model_set

        self.reg_huber = HuberRegressor(epsilon=huber_epsilon, max_iter=1000)
        self.reg_ridge = Ridge(alpha=10.0)
        self.reg_rf    = RandomForestRegressor(
            n_estimators=120, max_depth=5, min_samples_leaf=4, random_state=42
        )

        if model_set == "xgb":
            from xgboost import XGBRegressor
            self.reg_hgb = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9,
                random_state=42, verbosity=0,
            )
        else:
            self.reg_hgb = HistGradientBoostingRegressor(
                max_iter=120, max_depth=4, learning_rate=0.04, random_state=42
            )

        self.clf = RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )

        if weights is None:
            weights = {'huber': 0.35, 'ridge': 0.25, 'hgb': 0.20, 'rf': 0.20}
        self.weights = weights

        # CI statistics (populated by train_and_save_model)
        self.residual_std       = 0.50     # OOF residual std
        self.residual_p10       = -0.641   # 10th percentile (Gaussian fallback)
        self.residual_p90       =  0.641   # 90th percentile
        self.train_residual_std = 0.50     # baseline for scaling

        self.feature_names = FEATURE_COLS
        self.metrics = {}

    # ------------------------------------------------------------------ fit --
    def fit(self, X: pd.DataFrame, y: pd.Series):
        Xm = X[self.feature_names].values
        yv = y.values

        self.reg_huber.fit(Xm, yv)
        self.reg_ridge.fit(Xm, yv)
        self.reg_hgb.fit(Xm, yv)
        self.reg_rf.fit(Xm, yv)

        y_dir = np.where(yv >  FLAT_BAND,  1,
                np.where(yv < -FLAT_BAND, -1, 0))
        self.clf.fit(Xm, y_dir)
        return self

    # -------------------------------------------------------------- predict --
    def predict_pct(self, X: pd.DataFrame) -> np.ndarray:
        Xm = X[self.feature_names].values
        p_h = self.reg_huber.predict(Xm)
        p_r = self.reg_ridge.predict(Xm)
        p_g = self.reg_hgb.predict(Xm)
        p_f = self.reg_rf.predict(Xm)
        return (self.weights['huber'] * p_h +
                self.weights['ridge'] * p_r +
                self.weights['hgb']   * p_g +
                self.weights['rf']    * p_f)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.feature_names].values)

    def predict_details(self, X_row: pd.DataFrame, last_close: float,
                        recent_residuals=None) -> dict:
        """
        recent_residuals : list/array of recent (actual - predicted) gaps.
                           If provided and len >= MIN_RESID_SAMPLES, the
                           CI band is scaled by their std / train_residual_std.
        """
        gap_pct    = float(self.predict_pct(X_row)[0])
        gap_points = round(gap_pct * last_close / 100.0, 1)
        pred_open  = round(last_close + gap_points, 1)

        # --- Volatility-scaled 80% CI ---
        vol_scale = 1.0
        if (recent_residuals is not None
                and len(recent_residuals) >= MIN_RESID_SAMPLES
                and self.train_residual_std > 0):
            recent_std = float(np.std(recent_residuals))
            raw_scale  = recent_std / self.train_residual_std
            vol_scale  = max(SCALE_MIN, min(SCALE_MAX, raw_scale))

        lo_pct = self.residual_p10 * vol_scale
        hi_pct = self.residual_p90 * vol_scale

        low_open  = round(last_close * (1 + (gap_pct + lo_pct) / 100.0), 1)
        high_open = round(last_close * (1 + (gap_pct + hi_pct) / 100.0), 1)
        margin_pct = round((hi_pct - lo_pct) / 2.0, 2)

        probs    = self.clf.predict_proba(X_row[self.feature_names].values)[0]
        classes  = list(self.clf.classes_)
        prob_map = dict(zip(classes, probs))
        prob_up   = round(float(prob_map.get( 1, 0.0)) * 100, 1)
        prob_flat = round(float(prob_map.get( 0, 0.0)) * 100, 1)
        prob_down = round(float(prob_map.get(-1, 0.0)) * 100, 1)

        top_class = classes[int(np.argmax(probs))]
        direction = {1: "GAP UP", -1: "GAP DOWN", 0: "FLAT"}[int(top_class)]

        coefs = self.reg_ridge.coef_
        contributions = {
            feat: {
                'value': round(float(X_row[feat].iloc[0]), 2),
                'contribution_pct': round(float(X_row[feat].iloc[0]) * c, 3),
            }
            for feat, c in zip(self.feature_names, coefs)
        }
        sorted_drivers = sorted(
            contributions.items(),
            key=lambda kv: abs(kv[1]['contribution_pct']),
            reverse=True,
        )

        return {
            'gap_pct': round(gap_pct, 2),
            'gap_points': gap_points,
            'last_close': round(last_close, 1),
            'predicted_open': pred_open,
            'expected_low': low_open,
            'expected_high': high_open,
            'direction': direction,
            'prob_up': prob_up,
            'prob_flat': prob_flat,
            'prob_down': prob_down,
            'drivers': sorted_drivers[:6],
            'confidence_band_pct': margin_pct,
            'vol_scale': round(vol_scale, 2),
            'lo_pct': round(lo_pct, 3),
            'hi_pct': round(hi_pct, 3),
        }


# ---------------------------------------------------------------------------
# Training with OOF residual stats
# ---------------------------------------------------------------------------

def train_and_save_model(dataset: pd.DataFrame,
                         model_path: str = None,
                         weights: dict = None,
                         huber_epsilon: float = 1.35,
                         model_set: str = "paper",
                         n_folds: int = 10,
                         min_train_frac: float = 0.40):
    """
    1. Walk-forward OOF predictions to get honest residual stats.
    2. Full refit with those stats baked in.
    3. Save to disk.
    """
    ensure_model_dir()
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "nifty_gap_ensemble.joblib")

    X = dataset[FEATURE_COLS].reset_index(drop=True)
    y = dataset['gap_pct'].reset_index(drop=True)
    n = len(X)

    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = max(1, remaining // n_folds)

    oof = np.full(n, np.nan)
    for k in range(n_folds):
        tr_end = min_train + k * fold_size
        te_end = tr_end + fold_size if k < n_folds - 1 else n
        if te_end <= tr_end:
            break
        m = GapPredictionEnsemble(
            weights=weights,
            huber_epsilon=huber_epsilon,
            model_set=model_set,
        ).fit(X.iloc[:tr_end], y.iloc[:tr_end])
        oof[tr_end:te_end] = m.predict_pct(X.iloc[tr_end:te_end])

    valid = ~np.isnan(oof)
    yv, pv = y.values[valid], oof[valid]
    cv = dataset['prev_close'].values[valid]

    mae_pct  = mean_absolute_error(yv, pv)
    rmse_pct = np.sqrt(mean_squared_error(yv, pv))
    r2       = r2_score(yv, pv)
    mae_pts  = mean_absolute_error(yv * cv / 100.0, pv * cv / 100.0)
    dir_acc  = float(np.mean(np.sign(yv) == np.sign(pv)) * 100)

    resid = yv - pv
    resid_std_oof = float(np.std(resid))
    resid_p10     = float(np.percentile(resid, 10))
    resid_p90     = float(np.percentile(resid, 90))

    lo = pv + resid_p10
    hi = pv + resid_p90
    coverage = float(np.mean((yv >= lo) & (yv <= hi)) * 100)

    metrics = {
        'test_samples': int(valid.sum()),
        'directional_accuracy_pct': round(dir_acc, 1),
        'mae_pct': round(float(mae_pct), 3),
        'rmse_pct': round(float(rmse_pct), 3),
        'mae_points': round(float(mae_pts), 1),
        'r2': round(float(r2), 3),
        'residual_std_oof': round(resid_std_oof, 4),
        'residual_p10': round(resid_p10, 4),
        'residual_p90': round(resid_p90, 4),
        'ci80_empirical_coverage_pct': round(coverage, 2),
    }

    # Full refit with honest residual stats
    full = GapPredictionEnsemble(
        weights=weights,
        huber_epsilon=huber_epsilon,
        model_set=model_set,
    ).fit(X, y)
    full.residual_std       = resid_std_oof
    full.residual_p10       = resid_p10
    full.residual_p90       = resid_p90
    full.train_residual_std = resid_std_oof   # baseline for scaling
    full.metrics = metrics

    joblib.dump(full, model_path)
    print(f"Model saved -> {model_path}")
    print(f"OOF  MAE={metrics['mae_pct']}%  DirAcc={metrics['directional_accuracy_pct']}%  "
          f"MAE_pts={metrics['mae_points']}  R2={metrics['r2']}  "
          f"sigma_OOF={metrics['residual_std_oof']}  "
          f"CI80_emp_cov={metrics['ci80_empirical_coverage_pct']}%")
    return full, metrics


def load_trained_model(model_path: str = None):
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "nifty_gap_ensemble.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)