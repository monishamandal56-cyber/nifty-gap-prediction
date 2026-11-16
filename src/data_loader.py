"""
src/data_loader.py
------------------
Fetches OHLCV data for the 11 instruments tracked by the Nifty 50 gap
prediction pipeline. Uses a file-based Parquet cache with a configurable TTL.
"""

import os
import time
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"
)
CACHE_TTL_SECONDS = 3600   # 1 hour

TICKERS = {
    'nifty':      '^NSEI',
    'inda':       'INDA',
    'sp500':      '^GSPC',
    'nasdaq':     '^IXIC',
    'dow':        '^DJI',
    'sp_futures': 'ES=F',
    'nq_futures': 'NQ=F',
    'nikkei':     '^N225',
    'vix':        '^INDIAVIX',
    'crude':      'CL=F',
    'usdinr':     'USDINR=X',
}

OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str, period: str, start, end) -> str:
    """Cache filename encodes period OR date range, so switching arguments
    never silently returns stale data."""
    safe = symbol.replace('^', '').replace('=', '_')
    if start is not None or end is not None:
        tag = f"{start or 'min'}_{end or 'max'}"
    else:
        tag = period
    return os.path.join(CACHE_DIR, f"{safe}_{tag}.parquet")


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures a flat column index, a naive DatetimeIndex, OHLCV columns, and
    no duplicate dates. Yahoo sometimes returns MultiIndex columns or
    duplicate rows after corporate actions.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)

    # Flatten MultiIndex columns like ('Close', '^NSEI') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Ensure required columns exist (Volume is missing for some indices)
    for col in OHLCV_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[OHLCV_COLS].copy()

    # Normalize the index: strip timezone, drop time-of-day
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"

    # Drop duplicates (Yahoo occasionally repeats a date after splits)
    df = df[~df.index.duplicated(keep='last')]

    # Drop rows where the OHLC block is missing (keep Volume NaN)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    return df.sort_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ticker_data(symbol: str,
                      period: str = "5y",
                      start: str = None,
                      end: str = None,
                      force_refresh: bool = False,
                      ttl_seconds: int = CACHE_TTL_SECONDS) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a symbol.

    Pass either period (relative, e.g. "5y") OR an explicit start/end
    date range (strings, e.g. "2020-01-01"). Explicit ranges are strongly
    recommended for reproducible research.

    - Uses adjusted prices (auto_adjust=True).
    - Caches to Parquet with a TTL; falls back to cache on network failure.
    - Raises RuntimeError if neither live data nor cache is available.
    """
    _ensure_cache_dir()
    cache_path = _cache_path(symbol, period, start, end)

    # 1. Cache hit?
    if not force_refresh and os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < ttl_seconds:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass  # fall through to download

    # 2. Download
    try:
        kwargs = dict(interval="1d", progress=False, auto_adjust=True)
        if start is not None or end is not None:
            kwargs.update(start=start, end=end)
        else:
            kwargs.update(period=period)

        raw = yf.download(symbol, **kwargs)
        df = _normalize_df(raw)

        if df.empty:
            raise RuntimeError(f"yfinance returned empty data for {symbol}")

        df.to_parquet(cache_path)
        return df

    except Exception as e:
        # 3. Fall back to whatever cache we have, even if stale
        if os.path.exists(cache_path):
            print(f"[data_loader] live fetch failed for {symbol} ({e}); using stale cache.")
            return pd.read_parquet(cache_path)
        raise RuntimeError(f"Failed to fetch {symbol} and no cache available: {e}")


def load_all_historical_data(period: str = "5y",
                             start: str = None,
                             end: str = None,
                             force_refresh: bool = False) -> dict:
    """
    Returns a dict of {ticker_key: DataFrame} for every ticker in TICKERS.
    Skips any ticker that raises; prints a warning so you notice.
    """
    data = {}
    for key, symbol in TICKERS.items():
        try:
            df = fetch_ticker_data(symbol, period=period,
                                   start=start, end=end,
                                   force_refresh=force_refresh)
            if not df.empty:
                data[key] = df
            else:
                print(f"[data_loader] warning: no data for {key} ({symbol})")
        except Exception as e:
            print(f"[data_loader] error for {key} ({symbol}): {e}")
    return data


def get_latest_market_snapshot() -> dict:
    """
    Lightweight live snapshot for the dashboard: last 5 daily bars per ticker.
    Not cached (deliberately - this is the live tape). Do not call in a loop.
    """
    snapshot = {}
    for key, symbol in TICKERS.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
            if hist.empty:
                continue
            last_close = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else last_close
            change_pct = (last_close - prev_close) / prev_close * 100
            snapshot[key] = {
                'symbol': symbol,
                'last_close': round(last_close, 2),
                'prev_close': round(prev_close, 2),
                'change_pct': round(change_pct, 2),
                'date': hist.index[-1].strftime('%Y-%m-%d'),
                'open': round(float(hist['Open'].iloc[-1]), 2),
                'high': round(float(hist['High'].iloc[-1]), 2),
                'low':  round(float(hist['Low'].iloc[-1]), 2),
            }
        except Exception as e:
            print(f"[data_loader] snapshot error for {symbol}: {e}")
    return snapshot


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing data loader (1-month window) ...")
    d = load_all_historical_data(period="1mo")
    for k, v in d.items():
        print(f"  {k:12s}: {len(v):4d} rows, last date {v.index[-1].date()}")
    print("\nLive snapshot:")
    for k, v in get_latest_market_snapshot().items():
        print(f"  {k:12s}: {v['last_close']:>10.2f}  ({v['change_pct']:+.2f}%)  on {v['date']}")