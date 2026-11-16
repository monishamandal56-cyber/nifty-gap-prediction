"""
Nifty 50 Next-Day Gap Prediction System.

Package modules:
    data_loader         -- OHLCV ingestion and Parquet caching
    feature_engineering -- Point-in-time feature construction and alignment
    models              -- Dual-head ML ensemble and walk-forward validation
    predictor           -- End-to-end inference orchestrator
    ablation            -- Walk-forward ablation study harness
"""

__all__ = [
    "data_loader",
    "feature_engineering",
    "models",
    "predictor",
    "ablation",
]
