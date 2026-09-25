from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sector Momentum Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Sector Momentum Pro 2.0"
CACHE_DIR = Path(".sector_momentum_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PRICE_CACHE_FILE = CACHE_DIR / "ohlcv_cache.parquet"
PIPELINE_LOG_FILE = CACHE_DIR / "pipeline_log.csv"
SIGNAL_STATE_FILE = CACHE_DIR / "signal_states.csv"
TRADING_DAYS = {"1M": 21, "3M": 63, "6M": 126}
LOOKBACKS = {
    "6 Months": (126, "2y"),
    "1 Year": (252, "2y"),
    "2 Years": (504, "5y"),
}

SECTOR_NORMALIZATION = {
    "Information Technology": "Technology",
    "Technology": "Technology",
    "Financials": "Financial Services",
    "Financial Services": "Financial Services",
    "Health Care": "Healthcare",
    "Healthcare": "Healthcare",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Cyclical": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Consumer Defensive": "Consumer Defensive",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Materials": "Basic Materials",
    "Basic Materials": "Basic Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}

HEATMAP_SCALE = [
    [0.00, "#450a0a"],
    [0.25, "#991b1b"],
    [0.49, "#374151"],
    [0.50, "#1f2937"],
    [0.75, "#166534"],
    [1.00, "#052e16"],
]

st.markdown(
    """
    <style>
    .stApp { background:#050b11; }
    [data-testid="stSidebar"] { background:#07121c; border-right:1px solid rgba(255,255,255,.08); }
    .block-container { max-width:1650px; padding-top:1.1rem; padding-bottom:3rem; }
    .terminal-title { font-size:2.35rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; }
    .terminal-subtitle { color:#94a3b8; font-size:.92rem; margin-top:6px; margin-bottom:16px; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.08); border-radius:10px; padding:12px 14px; }
    div[data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.08); border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_ticker(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper().replace(".", "-")


def normalize_sector(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    value = str(value).strip()
    return SECTOR_NORMALIZATION.get(value, value)


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def trailing_return(series: pd.Series, periods: int) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    start, end = s.iloc[-periods - 1], s.iloc[-1]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return np.nan
    return (end / start - 1.0) * 100.0


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / sd


def pct_rank(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(pct=True, method="average") * 100.0


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return out


def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0].rename(
        columns={
            "Symbol": "Ticker",
            "Security": "Company",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "Industry",
        }
    )
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    df["Sector"] = df["Sector"].map(normalize_sector)
    df["SP500"] = True
    df["NASDAQ100"] = False
    return df[["Ticker", "Company", "Sector", "Industry", "SP500", "NASDAQ100"]]


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_nasdaq100() -> pd.DataFrame:
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    candidate: Optional[pd.DataFrame] = None
    for table in tables:
        cols = {str(c).strip().lower() for c in table.columns}
        has_ticker = bool(cols.intersection({"ticker", "ticker symbol", "symbol"}))
        has_company = bool(cols.intersection({"company", "company name", "security"}))
        if has_ticker and has_company and len(table) >= 90:
            candidate = table.copy()
            break
    if candidate is None:
        raise RuntimeError("Nasdaq-100 table could not be identified")

    rename: Dict[object, str] = {}
    for col in candidate.columns:
        key = str(col).strip().lower()
        if key in {"ticker", "ticker symbol", "symbol"}:
            rename[col] = "Ticker"
        elif key in {"company", "company name", "security"}:
            rename[col] = "Company"
        elif key in {"gics sector", "sector"}:
            rename[col] = "Sector"
        elif key in {"gics sub-industry", "sub-industry", "industry"}:
            rename[col] = "Industry"

    candidate = candidate.rename(columns=rename)
    if "Ticker" not in candidate.columns:
        raise RuntimeError("Nasdaq-100 ticker column unavailable")
    if "Company" not in candidate.columns:
        candidate["Company"] = candidate["Ticker"]
    if "Sector" not in candidate.columns:
        candidate["Sector"] = "Unknown"
    if "Industry" not in candidate.columns:
        candidate["Industry"] = ""

    candidate["Ticker"] = candidate["Ticker"].map(normalize_ticker)
    candidate["Sector"] = candidate["Sector"].map(normalize_sector)
    candidate["SP500"] = False
    candidate["NASDAQ100"] = True
    return candidate[["Ticker", "Company", "Sector", "Industry", "SP500", "NASDAQ100"]]


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def build_universe() -> pd.DataFrame:
    sp = fetch_sp500()
    try:
        ndx = fetch_nasdaq100()
    except Exception:
        ndx = pd.DataFrame(columns=sp.columns)

    combined = pd.concat([sp, ndx], ignore_index=True)
    rows: List[dict] = []

    for ticker, group in combined.groupby("Ticker"):
        sectors = (
            group["Sector"].replace({"Unknown": np.nan, "": np.nan}).dropna().astype(str).tolist()
        )
        companies = group["Company"].dropna().astype(str).tolist()
        industries = group["Industry"].dropna().astype(str).tolist()
        rows.append(
            {
                "Ticker": ticker,
                "Company": companies[0] if companies else ticker,
                "Sector": normalize_sector(sectors[0] if sectors else "Unknown"),
                "Industry": industries[0] if industries else "",
                "SP500": bool(group["SP500"].any()),
                "NASDAQ100": bool(group["NASDAQ100"].any()),
            }
        )

    return pd.DataFrame(rows).drop_duplicates("Ticker").reset_index(drop=True)


def _ensure_multiindex(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        return data
    data = data.copy()
    data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])
    return data


@st.cache_data(ttl=60 * 20, show_spinner=False)
def download_prices(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    """Persistent Parquet cache + retrying yfinance downloader."""
    tickers = tuple(sorted({t for t in tickers if t}))
    cached = pd.DataFrame()

    if PRICE_CACHE_FILE.exists():
        try:
            cached = pd.read_parquet(PRICE_CACHE_FILE)
            if not isinstance(cached.columns, pd.MultiIndex):
                cached = pd.DataFrame()
        except Exception:
            cached = pd.DataFrame()

    batches: List[pd.DataFrame] = []
    logs = []

    for start in range(0, len(tickers), 80):
        batch = list(tickers[start:start + 80])
        data = pd.DataFrame()
        err = ""

        for attempt in range(3):
            try:
                data = yf.download(
                    tickers=batch,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=True,
                    group_by="column",
                    timeout=30,
                )
                if data is not None and not data.empty:
                    break
            except Exception as exc:
                err = str(exc)
            time.sleep(1.5 * (2 ** attempt))

        if data is None or data.empty:
            logs.append({
                "Timestamp": datetime.utcnow().isoformat(),
                "Batch": start // 80 + 1,
                "Requested": len(batch),
                "Returned": 0,
                "Status": "FAILED",
                "Error": err or "No data returned",
            })
            continue

        if len(batch) == 1:
            data = _ensure_multiindex(data, batch[0])
        batches.append(data)

        returned = len(set(data.columns.get_level_values(1))) if isinstance(data.columns, pd.MultiIndex) else 1
        logs.append({
            "Timestamp": datetime.utcnow().isoformat(),
            "Batch": start // 80 + 1,
            "Requested": len(batch),
            "Returned": returned,
            "Status": "OK",
            "Error": "",
        })

    fresh = pd.concat(batches, axis=1) if batches else pd.DataFrame()

    if not fresh.empty:
        fresh = fresh.loc[:, ~fresh.columns.duplicated()]
        result = fresh.combine_first(cached) if not cached.empty else fresh
        result = result.loc[:, ~result.columns.duplicated()].sort_index()
        try:
            result.to_parquet(PRICE_CACHE_FILE)
        except Exception:
            pass
    else:
        result = cached

    if logs:
        log_df = pd.DataFrame(logs)
        if PIPELINE_LOG_FILE.exists():
            try:
                old = pd.read_csv(PIPELINE_LOG_FILE)
                log_df = pd.concat([old, log_df], ignore_index=True).tail(2000)
            except Exception:
                pass
        try:
            log_df.to_csv(PIPELINE_LOG_FILE, index=False)
        except Exception:
            pass

    return result


def load_pipeline_log() -> pd.DataFrame:
    try:
        return pd.read_csv(PIPELINE_LOG_FILE) if PIPELINE_LOG_FILE.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def field(data: pd.DataFrame, name: str) -> pd.DataFrame:
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        return pd.DataFrame()
    if name not in data.columns.get_level_values(0):
        return pd.DataFrame()
    out = data[name].copy()
    if isinstance(out, pd.Series):
        out = out.to_frame()
    out.columns = [normalize_ticker(c) for c in out.columns]
    return out


def build_sector_indices(close_df: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    sector_series: Dict[str, pd.Series] = {}
    for sector in sorted(universe["Sector"].dropna().unique()):
        if sector == "Unknown":
            continue
        members = universe.loc[universe["Sector"] == sector, "Ticker"].tolist()
        members = [t for t in members if t in close_df.columns]
        if not members:
            continue

        rets = close_df[members].pct_change(fill_method=None)
        min_cov = max(1, math.ceil(len(members) * 0.40))
        coverage = rets.notna().sum(axis=1)
        ew = rets.mean(axis=1, skipna=True).where(coverage >= min_cov)
        sector_series[sector] = (1 + ew.fillna(0)).cumprod() * 100.0

    return pd.DataFrame(sector_series)


def sector_breadth(
    sector: str,
    universe: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> Dict[str, float]:
    members = universe.loc[universe["Sector"] == sector, "Ticker"].tolist()
    members = [t for t in members if t in close_df.columns]

    a21: List[float] = []
    a50: List[float] = []
    a200: List[float] = []
    highs: List[float] = []
    rvols: List[float] = []

    for ticker in members:
        close = close_df[ticker].dropna()
        if len(close) < 22:
            continue
        p = close.iloc[-1]
        a21.append(float(p > close.ewm(span=21, adjust=False).mean().iloc[-1]))
        if len(close) >= 50:
            a50.append(float(p > close.rolling(50).mean().iloc[-1]))
        if len(close) >= 200:
            a200.append(float(p > close.rolling(200).mean().iloc[-1]))
        prev_high = close.shift(1).rolling(20).max().iloc[-1]
        if not pd.isna(prev_high):
            highs.append(float(p > prev_high))

        if ticker in volume_df.columns:
            vol = volume_df[ticker].dropna()
            if len(vol) >= 21:
                avg = vol.iloc[-21:-1].mean()
                if avg > 0:
                    rvols.append(vol.iloc[-1] / avg)

    return {
        "Members": len(members),
        "Breadth >21D %": np.mean(a21) * 100 if a21 else np.nan,
        "Breadth >50D %": np.mean(a50) * 100 if a50 else np.nan,
        "Breadth >200D %": np.mean(a200) * 100 if a200 else np.nan,
        "20D High Breadth %": np.mean(highs) * 100 if highs else np.nan,
        "Median RVOL": np.median(rvols) if rvols else np.nan,
    }


def sector_scores(
    sector_indices: pd.DataFrame,
    benchmark: pd.Series,
    universe: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> pd.DataFrame:
    bench = {h: trailing_return(benchmark, d) for h, d in TRADING_DAYS.items()}
    rows: List[dict] = []

    for sector in sector_indices.columns:
        px = sector_indices[sector].dropna()
        perf = {h: trailing_return(px, d) for h, d in TRADING_DAYS.items()}
        rel = {
            h: perf[h] - bench[h]
            if not pd.isna(perf[h]) and not pd.isna(bench[h])
            else np.nan
            for h in TRADING_DAYS
        }
        rows.append(
            {
                "Sector": sector,
                "Return 1M %": perf["1M"],
                "Return 3M %": perf["3M"],
                "Return 6M %": perf["6M"],
                "RS 1M %": rel["1M"],
                "RS 3M %": rel["3M"],
                "RS 6M %": rel["6M"],
                **sector_breadth(sector, universe, close_df, volume_df),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for h in ["1M", "3M", "6M"]:
        df[f"Z {h}"] = zscore(df[f"RS {h} %"])

    df["Raw RS"] = 0.50 * df["Z 1M"] + 0.30 * df["Z 3M"] + 0.20 * df["Z 6M"]
    df["RS Score"] = pct_rank(df["Raw RS"])
    df["Breadth Score"] = (
        0.40 * df["Breadth >21D %"].fillna(50)
        + 0.30 * df["Breadth >50D %"].fillna(50)
        + 0.20 * df["Breadth >200D %"].fillna(50)
        + 0.10 * df["20D High Breadth %"].fillna(0)
    )
    df["Volume Score"] = pct_rank(zscore(df["Median RVOL"]))
    df["Momentum Score"] = (
        0.72 * df["RS Score"] + 0.20 * df["Breadth Score"] + 0.08 * df["Volume Score"]
    ).clip(0, 100)
    df["Rank"] = df["Momentum Score"].rank(ascending=False, method="min").astype(int)
    df["Leadership"] = np.select(
        [
            df["Momentum Score"] >= 80,
            df["Momentum Score"] >= 65,
            df["Momentum Score"] >= 45,
            df["Momentum Score"] >= 25,
        ],
        ["Leading", "Improving", "Neutral", "Weakening"],
        default="Lagging",
    )
    return df.sort_values("Momentum Score", ascending=False).reset_index(drop=True)


def weak_day_strength(stock: pd.Series, benchmark: pd.Series, lookback: int = 63) -> float:
    aligned = pd.concat([stock.rename("s"), benchmark.rename("b")], axis=1).dropna().tail(lookback)
    if len(aligned) < 22:
        return np.nan
    rets = aligned.pct_change(fill_method=None).dropna()
    weak = rets[rets["b"] < 0]
    if weak.empty:
        return np.nan
    return (weak["s"] - weak["b"]).mean() * 100.0


def classify_signal(
    price: float,
    rsi_value: float,
    ema8: float,
    ema21: float,
    sma50: float,
    sma200: float,
    macd_line: float,
    macd_signal: float,
    rvol: float,
    breakout: bool,
    ret_1m: float,
) -> str:
    strong_stack = (
        not pd.isna(sma50)
        and price > ema8 > ema21 > sma50
        and (pd.isna(sma200) or sma50 > sma200)
    )
    bullish = not pd.isna(sma50) and price > sma50 and (pd.isna(sma200) or sma50 > sma200)
    lagging = not pd.isna(sma50) and price < sma50 and (pd.isna(sma200) or sma50 < sma200)

    if breakout and strong_stack and (pd.isna(rvol) or rvol >= 1.25) and macd_line > macd_signal:
        return "Bullish Breakout"
    if not pd.isna(rsi_value) and rsi_value >= 75 and bullish:
        return "Overbought"
    if strong_stack and macd_line > macd_signal:
        return "Strong Uptrend"
    if bullish and ret_1m > 0 and macd_line > macd_signal:
        return "Bullish"
    if lagging and macd_line < macd_signal:
        return "Lagging"
    if not pd.isna(rsi_value) and rsi_value <= 30:
        return "Oversold"
    return "Neutral"


def stock_scores(
    universe: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    spy: pd.Series,
    qqq: pd.Series,
    ranking_horizon: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    days = TRADING_DAYS[ranking_horizon]

    bench = {
        "spy1": trailing_return(spy, 21),
        "spy3": trailing_return(spy, 63),
        "qqq1": trailing_return(qqq, 21),
    }

    for _, sec in universe.iterrows():
        ticker = sec["Ticker"]
        if ticker not in close_df.columns:
            continue

        close = close_df[ticker].dropna()
        if len(close) < 30:
            continue

        price = safe_float(close.iloc[-1])
        ret1 = trailing_return(close, 21)
        ret3 = trailing_return(close, 63)
        ret6 = trailing_return(close, 126)
        selected_ret = trailing_return(close, days)

        ema8 = safe_float(close.ewm(span=8, adjust=False).mean().iloc[-1])
        ema21 = safe_float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        sma50 = safe_float(close.rolling(50).mean().iloc[-1])
        sma200 = safe_float(close.rolling(200).mean().iloc[-1])

        rsi_value = safe_float(rsi(close).iloc[-1])
        macd_line_s, macd_signal_s, macd_hist_s = macd(close)
        macd_line = safe_float(macd_line_s.iloc[-1])
        macd_signal = safe_float(macd_signal_s.iloc[-1])
        macd_hist = safe_float(macd_hist_s.iloc[-1])

        volume = np.nan
        avg_vol = np.nan
        rvol = np.nan
        if ticker in volume_df.columns:
            vol = volume_df[ticker].dropna()
            if not vol.empty:
                volume = safe_float(vol.iloc[-1])
            if len(vol) >= 21:
                avg_vol = vol.iloc[-21:-1].mean()
                if avg_vol > 0:
                    rvol = volume / avg_vol

        prev_high = close.shift(1).rolling(20).max().iloc[-1]
        breakout = bool(not pd.isna(prev_high) and price > prev_high)

        dist50 = (price / sma50 - 1) * 100 if not pd.isna(sma50) and sma50 != 0 else np.nan
        dist200 = (price / sma200 - 1) * 100 if not pd.isna(sma200) and sma200 != 0 else np.nan
        wds = weak_day_strength(close, spy)

        rows.append(
            {
                "Ticker": ticker,
                "Company": sec["Company"],
                "Sector": sec["Sector"],
                "Industry": sec["Industry"],
                "Price": price,
                "Selected Return %": selected_ret,
                "1M Return %": ret1,
                "3M Return %": ret3,
                "6M Return %": ret6,
                "RS SPY 1M %": ret1 - bench["spy1"] if not pd.isna(ret1) and not pd.isna(bench["spy1"]) else np.nan,
                "RS SPY 3M %": ret3 - bench["spy3"] if not pd.isna(ret3) and not pd.isna(bench["spy3"]) else np.nan,
                "RS QQQ 1M %": ret1 - bench["qqq1"] if not pd.isna(ret1) and not pd.isna(bench["qqq1"]) else np.nan,
                "RSI 14": rsi_value,
                "EMA 8": ema8,
                "EMA 21": ema21,
                "SMA 50": sma50,
                "SMA 200": sma200,
                "Distance 50D %": dist50,
                "Distance 200D %": dist200,
                "MACD": macd_line,
                "MACD Signal": macd_signal,
                "MACD Histogram": macd_hist,
                "MACD State": "Bullish" if macd_line > macd_signal else "Bearish",
                "Volume": volume,
                "Avg Volume 20D": avg_vol,
                "RVOL": rvol,
                "20D Breakout": breakout,
                "Weak Day Strength %": wds,
                "Trend Signal": classify_signal(
                    price,
                    rsi_value,
                    ema8,
                    ema21,
                    sma50,
                    sma200,
                    macd_line,
                    macd_signal,
                    rvol,
                    breakout,
                    ret1,
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    factors = {
        "Selected Return %": "z_ret",
        "RS SPY 1M %": "z_spy1",
        "RS SPY 3M %": "z_spy3",
        "RS QQQ 1M %": "z_qqq1",
        "Distance 50D %": "z_d50",
        "Weak Day Strength %": "z_wds",
        "RVOL": "z_rvol",
    }
    for src, dst in factors.items():
        df[dst] = zscore(df[src])

    df["Raw Stock Momentum"] = (
        0.28 * df["z_ret"]
        + 0.20 * df["z_spy1"]
        + 0.15 * df["z_spy3"]
        + 0.12 * df["z_qqq1"]
        + 0.10 * df["z_d50"]
        + 0.10 * df["z_wds"]
        + 0.05 * df["z_rvol"]
    )
    df["Stock Momentum Score"] = pct_rank(df["Raw Stock Momentum"])
    df["Stock Rank"] = df["Stock Momentum Score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("Stock Momentum Score", ascending=False).reset_index(drop=True)


def market_regime(spy: pd.Series, qqq: pd.Series) -> Tuple[str, float, int]:
    def points(series: pd.Series) -> int:
        s = series.dropna()
        if len(s) < 200:
            return 0
        p = s.iloc[-1]
        ema21 = s.ewm(span=21, adjust=False).mean().iloc[-1]
        sma50 = s.rolling(50).mean().iloc[-1]
        sma200 = s.rolling(200).mean().iloc[-1]
        return int(p > ema21) + int(p > sma50) + int(sma50 > sma200) + int(trailing_return(s, 21) > 0)

    score = points(spy) + points(qqq)
    if score >= 7:
        return "RISK ON", 1.0, score
    if score >= 4:
        return "NEUTRAL", 0.5, score
    return "RISK OFF", 0.2, score


def stock_history(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = pd.DataFrame()
    for name in ["Open", "High", "Low", "Close", "Volume"]:
        matrix = field(raw, name)
        if ticker in matrix.columns:
            out[name] = matrix[ticker]

    if out.empty or "Close" not in out.columns:
        return pd.DataFrame()

    out = out.dropna(subset=["Close"]).copy()
    out["EMA8"] = out["Close"].ewm(span=8, adjust=False).mean()
    out["EMA21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    out["RSI"] = rsi(out["Close"])
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"])
    return out


def sector_treemap(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["Size"] = data["Members"].fillna(1).clip(lower=1)
    fig = px.treemap(
        data,
        path=["Sector"],
        values="Size",
        color="Momentum Score",
        color_continuous_scale=HEATMAP_SCALE,
        range_color=(0, 100),
        hover_data={
            "Momentum Score": ":.1f",
            "RS 1M %": ":.2f",
            "RS 3M %": ":.2f",
            "RS 6M %": ":.2f",
            "Breadth >50D %": ":.1f",
        },
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>Score %{color:.0f}",
        marker=dict(line=dict(color="#050b11", width=3)),
    )
    fig.update_layout(
        height=520,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="#050b11",
        plot_bgcolor="#050b11",
        font=dict(color="#e5e7eb"),
    )
    return fig


def stock_chart(history: pd.DataFrame, ticker: str, display_days: int) -> go.Figure:
    data = history.tail(display_days).copy()
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.52, 0.16, 0.17, 0.15],
        subplot_titles=(f"{ticker} Price", "Volume", "RSI (14)", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=ticker,
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    for label, col, color, width in [
        ("EMA 8", "EMA8", "#f8fafc", 1.0),
        ("EMA 21", "EMA21", "#a78bfa", 1.2),
        ("SMA 50", "SMA50", "#38bdf8", 1.5),
        ("SMA 200", "SMA200", "#f59e0b", 1.7),
    ]:
        fig.add_trace(
            go.Scatter(x=data.index, y=data[col], name=label, line=dict(color=color, width=width)),
            row=1,
            col=1,
        )

    if "Volume" in data.columns:
        colors = np.where(data["Close"] >= data["Open"], "#16a34a", "#dc2626")
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color=colors, opacity=0.65),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["RSI"], name="RSI 14", line=dict(color="#c084fc", width=1.6)),
        row=3,
        col=1,
    )
    for y, color in [(70, "#ef4444"), (50, "#64748b"), (30, "#22c55e")]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, opacity=0.6, row=3, col=1)

    fig.add_trace(
        go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#38bdf8", width=1.4)),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["MACD_SIGNAL"], name="MACD Signal", line=dict(color="#f59e0b", width=1.2)),
        row=4,
        col=1,
    )
    hcolors = np.where(data["MACD_HIST"] >= 0, "#16a34a", "#dc2626")
    fig.add_trace(
        go.Bar(x=data.index, y=data["MACD_HIST"], name="MACD Histogram", marker_color=hcolors, opacity=0.55),
        row=4,
        col=1,
    )

    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_layout(
        height=970,
        template="plotly_dark",
        paper_bgcolor="#050b11",
        plot_bgcolor="#050b11",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig



# =============================================================================
# PRO ENHANCEMENTS
# =============================================================================

def annualized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change(fill_method=None).rolling(window).std() * np.sqrt(252) * 100


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def data_quality_report(
    universe: pd.DataFrame,
    close_df: pd.DataFrame,
    min_history: int = 220,
    max_missing_pct: float = 15.0,
    stale_days: int = 4,
) -> pd.DataFrame:
    latest = close_df.index.max()
    rows = []
    for _, sec in universe.iterrows():
        ticker = sec["Ticker"]
        if ticker not in close_df.columns:
            rows.append({
                "Ticker": ticker, "Sector": sec["Sector"], "Status": "MISSING",
                "History": 0, "Missing %": 100.0, "Last Date": pd.NaT,
                "Outlier Flag": False, "Reason": "No market history",
            })
            continue

        series = close_df[ticker]
        valid = series.dropna()
        history = len(valid)
        missing = float(series.isna().mean() * 100)
        last_date = valid.index.max() if history else pd.NaT
        stale = bool(pd.notna(last_date) and (latest.normalize() - pd.Timestamp(last_date).normalize()).days > stale_days)
        invalid = bool((valid <= 0).any()) if history else True
        outlier = bool((valid.pct_change(fill_method=None).abs() > 0.60).any()) if history > 2 else False

        status, reasons = "OK", []
        if history < min_history:
            status, reasons = "INCOMPLETE", [f"<{min_history} observations"]
        if missing > max_missing_pct:
            status = "INCOMPLETE"
            reasons.append("high missing-data ratio")
        if stale:
            status = "STALE"
            reasons.append("stale latest bar")
        if invalid:
            status = "INVALID"
            reasons.append("non-positive close")

        rows.append({
            "Ticker": ticker, "Sector": sec["Sector"], "Status": status,
            "History": history, "Missing %": missing, "Last Date": last_date,
            "Outlier Flag": outlier, "Reason": ", ".join(reasons) if reasons else "Passed core checks",
        })
    return pd.DataFrame(rows)


def market_breadth_pro(close_df: pd.DataFrame, tickers: List[str]) -> Tuple[float, float]:
    above50, above200 = [], []
    for ticker in tickers:
        if ticker not in close_df.columns:
            continue
        c = close_df[ticker].dropna()
        if len(c) >= 50:
            above50.append(float(c.iloc[-1] > c.rolling(50).mean().iloc[-1]))
        if len(c) >= 200:
            above200.append(float(c.iloc[-1] > c.rolling(200).mean().iloc[-1]))
    b50 = np.mean(above50) * 100 if above50 else np.nan
    b200 = np.mean(above200) * 100 if above200 else np.nan
    return b50, b200


def market_regime_pro(
    spy: pd.Series,
    qqq: pd.Series,
    breadth50: float,
    breadth200: float,
    vix: Optional[pd.Series] = None,
) -> Tuple[str, float, int, float]:
    def points(series: pd.Series) -> int:
        s = series.dropna()
        if len(s) < 200:
            return 0
        p = s.iloc[-1]
        ema21 = s.ewm(span=21, adjust=False).mean().iloc[-1]
        sma50 = s.rolling(50).mean().iloc[-1]
        sma200 = s.rolling(200).mean().iloc[-1]
        return int(p > ema21) + int(p > sma50) + int(sma50 > sma200) + int(trailing_return(s, 21) > 0)

    score = points(spy) + points(qqq)
    if not pd.isna(breadth50):
        score += int(breadth50 >= 55)
    if not pd.isna(breadth200):
        score += int(breadth200 >= 50)

    vix_value = np.nan
    if vix is not None and len(vix.dropna()):
        vix_value = safe_float(vix.dropna().iloc[-1])
        if vix_value < 20:
            score += 1
        elif vix_value > 30:
            score -= 1
    else:
        spy_vol = safe_float(annualized_volatility(spy, 20).iloc[-1])
        if not pd.isna(spy_vol):
            score += int(spy_vol < 18)
            score -= int(spy_vol > 30)

    if score >= 9:
        return "RISK ON", 1.0, score, vix_value
    if score >= 5:
        return "NEUTRAL", 0.55, score, vix_value
    return "RISK OFF", 0.20, score, vix_value


def regime_weights(regime: str) -> Dict[str, float]:
    if regime == "RISK ON":
        return {
            "Selected Return %": 0.18, "Vol Adj Momentum": 0.14,
            "RS SPY 1M %": 0.15, "RS SPY 3M %": 0.10,
            "RS QQQ 1M %": 0.08, "Distance 50D %": 0.10,
            "Weak Day Strength %": 0.07, "RVOL": 0.08,
            "Persistence 21D": 0.10,
        }
    if regime == "NEUTRAL":
        return {
            "Selected Return %": 0.12, "Vol Adj Momentum": 0.17,
            "RS SPY 1M %": 0.16, "RS SPY 3M %": 0.14,
            "RS QQQ 1M %": 0.08, "Distance 50D %": 0.08,
            "Weak Day Strength %": 0.10, "RVOL": 0.05,
            "Persistence 21D": 0.10,
        }
    return {
        "Selected Return %": 0.07, "Vol Adj Momentum": 0.20,
        "RS SPY 1M %": 0.18, "RS SPY 3M %": 0.15,
        "RS QQQ 1M %": 0.07, "Distance 50D %": 0.05,
        "Weak Day Strength %": 0.15, "RVOL": 0.03,
        "Persistence 21D": 0.10,
    }


def sector_score_history(
    sector_idx: pd.DataFrame,
    benchmark: pd.Series,
    lookback_dates: int = 30,
) -> pd.DataFrame:
    joined = sector_idx.join(benchmark.rename("BENCH"), how="inner")
    records = []
    for dt in joined.index[-lookback_dates:]:
        hist = joined.loc[:dt]
        if len(hist) < 127:
            continue
        bench = hist["BENCH"].dropna()
        rows = []
        for sector in sector_idx.columns:
            s = hist[sector].dropna()
            if len(s) < 127:
                continue
            rows.append({
                "Sector": sector,
                "RS1": trailing_return(s, 21) - trailing_return(bench, 21),
                "RS3": trailing_return(s, 63) - trailing_return(bench, 63),
                "RS6": trailing_return(s, 126) - trailing_return(bench, 126),
            })
        x = pd.DataFrame(rows)
        if x.empty:
            continue
        x["Raw"] = 0.50 * zscore(x["RS1"]) + 0.30 * zscore(x["RS3"]) + 0.20 * zscore(x["RS6"])
        x["Score"] = pct_rank(x["Raw"])
        x["Rank"] = x["Score"].rank(ascending=False, method="min")
        for _, r in x.iterrows():
            records.append({"Date": dt, "Sector": r["Sector"], "Score": r["Score"], "Rank": r["Rank"]})
    return pd.DataFrame(records)


def add_sector_rotation(metrics: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    if history.empty:
        out["Score Change 5D"] = np.nan
        out["Score Change 10D"] = np.nan
        out["Rank Change 5D"] = np.nan
        out["Rotation Status"] = "N/A"
        return out

    dates = sorted(history["Date"].unique())
    latest = history[history["Date"] == dates[-1]].set_index("Sector")
    prev5 = history[history["Date"] == dates[-6]].set_index("Sector") if len(dates) >= 6 else pd.DataFrame()
    prev10 = history[history["Date"] == dates[-11]].set_index("Sector") if len(dates) >= 11 else pd.DataFrame()

    d5, d10, r5 = [], [], []
    for sector in out["Sector"]:
        cs = safe_float(latest.loc[sector, "Score"]) if sector in latest.index else np.nan
        cr = safe_float(latest.loc[sector, "Rank"]) if sector in latest.index else np.nan
        s5 = safe_float(prev5.loc[sector, "Score"]) if not prev5.empty and sector in prev5.index else np.nan
        s10 = safe_float(prev10.loc[sector, "Score"]) if not prev10.empty and sector in prev10.index else np.nan
        rr5 = safe_float(prev5.loc[sector, "Rank"]) if not prev5.empty and sector in prev5.index else np.nan
        d5.append(cs - s5 if not pd.isna(cs) and not pd.isna(s5) else np.nan)
        d10.append(cs - s10 if not pd.isna(cs) and not pd.isna(s10) else np.nan)
        r5.append(rr5 - cr if not pd.isna(rr5) and not pd.isna(cr) else np.nan)

    out["Score Change 5D"] = d5
    out["Score Change 10D"] = d10
    out["Rank Change 5D"] = r5
    out["Rotation Status"] = np.select(
        [
            (out["Momentum Score"] >= 70) & (out["Score Change 5D"] >= 8),
            (out["Momentum Score"] >= 70) & (out["Score Change 5D"] > -5),
            out["Score Change 5D"] <= -8,
            out["Momentum Score"] < 40,
        ],
        ["ACCELERATING", "LEADING", "DECELERATING", "LAGGING"],
        default="STABLE",
    )
    return out


def simplified_stock_history(
    close_df: pd.DataFrame,
    benchmark: pd.Series,
    tickers: List[str],
    lookback_days: int = 30,
) -> pd.DataFrame:
    tickers = [t for t in tickers if t in close_df.columns]
    if not tickers:
        return pd.DataFrame()

    c = close_df[tickers]
    r21 = c.pct_change(21, fill_method=None)
    r63 = c.pct_change(63, fill_method=None)
    rs21 = r21.sub(benchmark.pct_change(21, fill_method=None), axis=0)
    rs63 = r63.sub(benchmark.pct_change(63, fill_method=None), axis=0)
    sma50 = c.rolling(50).mean()
    d50 = c / sma50 - 1
    vol20 = c.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)
    va = r63 / vol20.replace(0, np.nan)

    records = []
    for dt in c.index[-lookback_days:]:
        frame = pd.DataFrame({
            "R21": r21.loc[dt], "R63": r63.loc[dt],
            "RS21": rs21.loc[dt], "RS63": rs63.loc[dt],
            "D50": d50.loc[dt], "VA": va.loc[dt],
        })
        raw = (
            0.20 * zscore(frame["R21"]) + 0.20 * zscore(frame["R63"])
            + 0.20 * zscore(frame["RS21"]) + 0.15 * zscore(frame["RS63"])
            + 0.10 * zscore(frame["D50"]) + 0.15 * zscore(frame["VA"])
        )
        score = pct_rank(raw)
        tmp = pd.DataFrame({"Ticker": score.index, "Score": score.values, "Date": dt})
        records.append(tmp)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def persistence_features(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["Ticker", "Persistence 5D", "Persistence 21D", "Days Top 10%", "Rank Change 5D"])

    dates = sorted(history["Date"].unique())
    last5, last21 = dates[-5:], dates[-21:]
    latest = history[history["Date"] == dates[-1]].set_index("Ticker")["Score"]
    prev_date = dates[-6] if len(dates) >= 6 else dates[0]
    prev = history[history["Date"] == prev_date].set_index("Ticker")["Score"]
    rows = []

    for ticker, g in history.groupby("Ticker"):
        p5 = g[g["Date"].isin(last5)]["Score"].mean()
        p21 = g[g["Date"].isin(last21)]["Score"].mean()
        top10 = int((g[g["Date"].isin(last21)]["Score"] >= 90).sum())
        cur = safe_float(latest.get(ticker, np.nan))
        old = safe_float(prev.get(ticker, np.nan))
        rows.append({
            "Ticker": ticker, "Persistence 5D": p5, "Persistence 21D": p21,
            "Days Top 10%": top10,
            "Rank Change 5D": cur - old if not pd.isna(cur) and not pd.isna(old) else np.nan,
        })
    return pd.DataFrame(rows)


def volume_confirmation(volume: pd.Series, close: pd.Series) -> Dict[str, object]:
    x = pd.concat([volume.rename("V"), close.rename("C")], axis=1).dropna()
    if len(x) < 30:
        return {"RVOL": np.nan, "Avg RVOL 5D": np.nan, "Up/Down Volume Ratio": np.nan, "RVOL Trend": np.nan, "Volume Confirmation": "INSUFFICIENT"}

    avg20 = x["V"].shift(1).rolling(20).mean()
    rvol_s = x["V"] / avg20.replace(0, np.nan)
    rvol = safe_float(rvol_s.iloc[-1])
    avg5 = safe_float(rvol_s.tail(5).mean())
    trend = safe_float(rvol_s.tail(5).mean() - rvol_s.iloc[-10:-5].mean())
    rets = x["C"].pct_change(fill_method=None)
    upv = x.loc[rets > 0, "V"].tail(20).sum()
    dnv = x.loc[rets < 0, "V"].tail(20).sum()
    ud = upv / dnv if dnv > 0 else np.nan

    if not pd.isna(rvol) and rvol >= 1.5 and trend > 0:
        label = "VOLUME CONFIRMED"
    elif not pd.isna(rvol) and rvol >= 1.2:
        label = "NORMAL+"
    elif not pd.isna(rvol) and rvol < 0.8:
        label = "WEAK"
    else:
        label = "NORMAL"

    return {"RVOL": rvol, "Avg RVOL 5D": avg5, "Up/Down Volume Ratio": ud, "RVOL Trend": trend, "Volume Confirmation": label}


def stock_scores_pro(
    universe: pd.DataFrame,
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    spy: pd.Series,
    qqq: pd.Series,
    sector_metrics: pd.DataFrame,
    persistence: pd.DataFrame,
    ranking_horizon: str,
    regime: str,
) -> pd.DataFrame:
    days = TRADING_DAYS[ranking_horizon]
    sector_map = sector_metrics.set_index("Sector")["Momentum Score"].to_dict()
    pmap = persistence.set_index("Ticker").to_dict("index") if not persistence.empty else {}
    bench = {
        "spy1": trailing_return(spy, 21), "spy3": trailing_return(spy, 63),
        "qqq1": trailing_return(qqq, 21),
    }
    rows = []

    for _, sec in universe.iterrows():
        ticker = sec["Ticker"]
        if ticker not in close_df.columns:
            continue
        c = close_df[ticker].dropna()
        if len(c) < 220:
            continue

        price = safe_float(c.iloc[-1])
        ret1, ret3, ret6 = trailing_return(c, 21), trailing_return(c, 63), trailing_return(c, 126)
        selected = trailing_return(c, days)
        ema8 = safe_float(c.ewm(span=8, adjust=False).mean().iloc[-1])
        ema21 = safe_float(c.ewm(span=21, adjust=False).mean().iloc[-1])
        sma50 = safe_float(c.rolling(50).mean().iloc[-1])
        sma200 = safe_float(c.rolling(200).mean().iloc[-1])
        rsi_value = safe_float(rsi(c).iloc[-1])
        macd_l_s, macd_s_s, macd_h_s = macd(c)
        macd_l, macd_s = safe_float(macd_l_s.iloc[-1]), safe_float(macd_s_s.iloc[-1])
        vol20 = safe_float(annualized_volatility(c, 20).iloc[-1])
        vol_adj = selected / vol20 if not pd.isna(selected) and not pd.isna(vol20) and vol20 != 0 else np.nan

        prev_high = c.shift(1).rolling(20).max().iloc[-1]
        breakout = bool(price > prev_high) if not pd.isna(prev_high) else False
        distance_breakout = (price / prev_high - 1) * 100 if not pd.isna(prev_high) and prev_high != 0 else np.nan
        high52 = c.rolling(252).max().iloc[-1]
        distance52 = (price / high52 - 1) * 100 if not pd.isna(high52) and high52 != 0 else np.nan

        atr14 = np.nan
        if ticker in high_df.columns and ticker in low_df.columns:
            hh = high_df[ticker].reindex(c.index)
            ll = low_df[ticker].reindex(c.index)
            atr14 = safe_float(atr(hh, ll, c).iloc[-1])
        atr_pct = atr14 / price * 100 if not pd.isna(atr14) and price > 0 else np.nan

        vf = volume_confirmation(volume_df[ticker], c) if ticker in volume_df.columns else {
            "RVOL": np.nan, "Avg RVOL 5D": np.nan, "Up/Down Volume Ratio": np.nan,
            "RVOL Trend": np.nan, "Volume Confirmation": "INSUFFICIENT",
        }
        p = pmap.get(ticker, {})
        sector_score = safe_float(sector_map.get(sec["Sector"], np.nan))

        rows.append({
            "Ticker": ticker, "Company": sec["Company"], "Sector": sec["Sector"], "Industry": sec["Industry"],
            "Price": price, "Selected Return %": selected, "1M Return %": ret1, "3M Return %": ret3, "6M Return %": ret6,
            "RS SPY 1M %": ret1 - bench["spy1"], "RS SPY 3M %": ret3 - bench["spy3"],
            "RS QQQ 1M %": ret1 - bench["qqq1"],
            "Realized Vol 20D %": vol20, "Vol Adj Momentum": vol_adj,
            "RSI 14": rsi_value, "EMA 8": ema8, "EMA 21": ema21, "SMA 50": sma50, "SMA 200": sma200,
            "Distance 50D %": (price / sma50 - 1) * 100 if sma50 else np.nan,
            "Distance 200D %": (price / sma200 - 1) * 100 if sma200 else np.nan,
            "Distance 52W High %": distance52,
            "MACD": macd_l, "MACD Signal": macd_s, "MACD Histogram": safe_float(macd_h_s.iloc[-1]),
            "MACD State": "Bullish" if macd_l > macd_s else "Bearish",
            "RVOL": safe_float(vf["RVOL"]), "Avg RVOL 5D": safe_float(vf["Avg RVOL 5D"]),
            "Up/Down Volume Ratio": safe_float(vf["Up/Down Volume Ratio"]), "RVOL Trend": safe_float(vf["RVOL Trend"]),
            "Volume Confirmation": vf["Volume Confirmation"],
            "Weak Day Strength %": weak_day_strength(c, spy), "20D Breakout": breakout,
            "Distance to Breakout %": distance_breakout, "ATR14": atr14, "ATR %": atr_pct,
            "Persistence 5D": safe_float(p.get("Persistence 5D", np.nan)),
            "Persistence 21D": safe_float(p.get("Persistence 21D", np.nan)),
            "Days Top 10%": safe_float(p.get("Days Top 10%", np.nan)),
            "Rank Change 5D": safe_float(p.get("Rank Change 5D", np.nan)),
            "Sector Momentum Score": sector_score,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    weights = regime_weights(regime)
    df["Raw Stock Momentum"] = 0.0
    for factor, weight in weights.items():
        zcol = f"Z::{factor}"
        ccol = f"Contrib::{factor}"
        df[zcol] = zscore(df[factor])
        df[ccol] = df[zcol] * weight
        df["Raw Stock Momentum"] += df[ccol].fillna(0)

    df["Stock Momentum Score"] = pct_rank(df["Raw Stock Momentum"])

    trend_ok = ((df["Price"] > df["SMA 50"]) & (df["SMA 50"] > df["SMA 200"])).astype(float)
    macd_ok = (df["MACD"] > df["MACD Signal"]).astype(float)
    sector_ok = (df["Sector Momentum Score"] >= 70).astype(float)
    rs_ok = (df["RS SPY 1M %"] > 0).astype(float)
    volume_ok = df["Volume Confirmation"].isin(["VOLUME CONFIRMED", "NORMAL+"]).astype(float)
    persistence_ok = (df["Persistence 21D"] >= 70).astype(float)
    df["Signal Confidence"] = 20*trend_ok + 15*macd_ok + 20*sector_ok + 15*rs_ok + 15*volume_ok + 15*persistence_ok

    df["Composite Leadership Score"] = (
        0.55 * df["Stock Momentum Score"] + 0.25 * df["Sector Momentum Score"].fillna(50) + 0.20 * df["Signal Confidence"]
    ).clip(0, 100)
    df["Stock Rank"] = df["Stock Momentum Score"].rank(ascending=False, method="min").astype(int)
    df["Composite Rank"] = df["Composite Leadership Score"].rank(ascending=False, method="min").astype(int)
    df["Persistence Badge"] = np.select(
        [
            (df["Persistence 21D"] >= 90) & (df["Days Top 10%"] >= 10),
            df["Rank Change 5D"] >= 12,
            df["Rank Change 5D"] <= -12,
        ],
        ["PERSISTENT LEADER", "NEW LEADER", "RANK DETERIORATING"],
        default="STABLE",
    )

    stack = (df["Price"] > df["EMA 8"]) & (df["EMA 8"] > df["EMA 21"]) & (df["EMA 21"] > df["SMA 50"]) & (df["SMA 50"] > df["SMA 200"])
    df["Trend Signal"] = np.select(
        [
            df["20D Breakout"] & stack & (df["RVOL"] >= 1.5) & (df["MACD"] > df["MACD Signal"]) & (df["Sector Momentum Score"] >= 70),
            (df["RSI 14"] >= 75) & (df["Price"] > df["SMA 50"]),
            stack & (df["MACD"] > df["MACD Signal"]),
            (df["Price"] > df["SMA 50"]) & (df["MACD"] > df["MACD Signal"]),
            (df["Price"] < df["SMA 50"]) & (df["MACD"] < df["MACD Signal"]),
        ],
        ["Bullish Breakout", "Overbought", "Strong Uptrend", "Bullish", "Lagging"],
        default="Neutral",
    )
    return df.sort_values("Composite Leadership Score", ascending=False).reset_index(drop=True)



def add_signal_trajectory(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if history.empty:
        out["Signal Age"] = np.nan
        out["Peak Historical Score"] = np.nan
        out["Score Decay"] = np.nan
        out["Score Change 10D"] = np.nan
        out["Momentum Trajectory"] = "UNKNOWN"
        return out

    rows = []
    for ticker, g in history.groupby("Ticker"):
        g = g.sort_values("Date")
        scores = g["Score"].dropna()
        if scores.empty:
            continue
        current = safe_float(scores.iloc[-1])
        ch5 = current - safe_float(scores.iloc[-5]) if len(scores) >= 5 else np.nan
        ch10 = current - safe_float(scores.iloc[-10]) if len(scores) >= 10 else np.nan
        peak = safe_float(scores.max())
        age = 0
        for value in reversed(scores.tolist()):
            if value >= 80:
                age += 1
            else:
                break
        if not pd.isna(ch5) and ch5 >= 10 and age <= 5:
            trajectory = "EMERGING"
        elif not pd.isna(ch5) and ch5 >= 5 and not pd.isna(ch10) and ch10 >= 8:
            trajectory = "ACCELERATING"
        elif not pd.isna(ch5) and ch5 <= -8:
            trajectory = "REVERSING"
        elif not pd.isna(ch5) and ch5 < -3:
            trajectory = "FADING"
        elif age >= 10:
            trajectory = "MATURE"
        else:
            trajectory = "STABLE"
        rows.append({
            "Ticker": ticker,
            "Signal Age": age,
            "Peak Historical Score": peak,
            "Score Decay": current - peak,
            "Score Change 10D": ch10,
            "Momentum Trajectory": trajectory,
        })
    return out.merge(pd.DataFrame(rows), on="Ticker", how="left")


def add_breakout_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ATR Extension"] = (out["Price"] - out["EMA 21"]) / out["ATR14"].replace(0, np.nan)
    proximity = (100 - out["Distance to Breakout %"].abs() * 12).clip(0, 100)
    volume_score = ((out["RVOL"].fillna(0) / 2.0) * 100).clip(0, 100)
    rs_score = percentile_rank(out["RS SPY 1M %"]).fillna(50)
    penalty = ((out["ATR Extension"].fillna(0) - 2.5).clip(lower=0) * 12).clip(0, 40)
    out["Breakout Quality"] = (
        0.30 * proximity
        + 0.25 * volume_score
        + 0.25 * out["Sector Momentum Score"].fillna(50)
        + 0.20 * rs_score
        - penalty
    ).clip(0, 100)
    return out


def load_signal_states() -> pd.DataFrame:
    try:
        return pd.read_csv(SIGNAL_STATE_FILE) if SIGNAL_STATE_FILE.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def update_signal_state_machine(df: pd.DataFrame) -> pd.DataFrame:
    previous = load_signal_states()
    prev = previous.set_index("Ticker").to_dict("index") if not previous.empty else {}
    records = []
    stamp = datetime.utcnow().isoformat()

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        prior = prev.get(ticker, {}).get("State", "WATCH")
        score = safe_float(row["Stock Momentum Score"], 0)
        conf = safe_float(row["Signal Confidence"], 0)
        quality = safe_float(row["Breakout Quality"], 0)
        dist = safe_float(row["Distance to Breakout %"], np.nan)
        rvol = safe_float(row["RVOL"], 0)
        breakout = bool(row["20D Breakout"])

        state, reason = "WATCH", "Monitoring"
        if prior == "ACTIVE" and (score < 65 or conf < 50):
            state, reason = "COOLDOWN", "Momentum or confidence deteriorated"
        elif prior == "COOLDOWN" and breakout and score >= 85 and quality >= 70:
            state, reason = "ARMED", "Fresh high-quality breakout after cooldown"
        elif prior == "CONFIRMED" and score >= 75:
            state, reason = "ACTIVE", "Confirmed setup remains strong"
        elif breakout and rvol >= 1.5 and quality >= 70 and conf >= 65:
            state, reason = "CONFIRMED", "Breakout confirmed by volume and quality"
        elif not pd.isna(dist) and -5 <= dist <= 0 and score >= 75:
            state, reason = "ARMED", "Near breakout with strong momentum"
        elif score < 60 or conf < 40:
            state, reason = "WATCH", "Below active threshold"

        records.append({"Ticker": ticker, "State": state, "State Reason": reason, "Last Update": stamp})

    states = pd.DataFrame(records)
    try:
        states.to_csv(SIGNAL_STATE_FILE, index=False)
    except Exception:
        pass
    return df.merge(states[["Ticker", "State", "State Reason"]], on="Ticker", how="left")


def correlation_matrix_60d(close_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    valid = [t for t in tickers if t in close_df.columns]
    if not valid:
        return pd.DataFrame()
    return close_df[valid].pct_change(fill_method=None).tail(60).corr()


def portfolio_risk_gate(
    candidates: pd.DataFrame,
    close_df: pd.DataFrame,
    portfolio_value: float,
    max_sector_exposure_pct: float,
    max_open_risk_pct: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), pd.DataFrame()

    out = candidates.sort_values("Composite Leadership Score", ascending=False).copy()
    active = out[out["State"].isin(["ACTIVE", "CONFIRMED"])].head(8)
    if active.empty:
        active = out.head(min(8, len(out)))

    corr = correlation_matrix_60d(close_df, list(dict.fromkeys(active["Ticker"].tolist() + out["Ticker"].head(50).tolist())))
    sector_exp = active.groupby("Sector")["Suggested Position $"].sum() / portfolio_value * 100
    open_risk_dollars = (
        (active["Price"] - active["Suggested Stop"]).clip(lower=0)
        * active["Suggested Shares"].fillna(0)
    ).sum()
    open_risk_pct = open_risk_dollars / portfolio_value * 100 if portfolio_value > 0 else np.nan

    decisions, max_corrs, reasons = [], [], []
    for _, row in out.iterrows():
        t, sector = row["Ticker"], row["Sector"]
        proposed = safe_float(row["Position %"], 0)
        current_sector = safe_float(sector_exp.get(sector, 0), 0)
        peers = [p for p in active["Ticker"] if p != t and t in corr.columns and p in corr.columns]
        max_corr = max([safe_float(corr.loc[t, p], -1) for p in peers], default=np.nan)

        if current_sector + proposed > max_sector_exposure_pct:
            decision, reason = "BLOCKED", "Sector exposure limit"
        elif not pd.isna(open_risk_pct) and open_risk_pct >= max_open_risk_pct:
            decision, reason = "BLOCKED", "Portfolio open-risk budget exhausted"
        elif not pd.isna(max_corr) and max_corr > 0.75:
            decision, reason = "REDUCE SIZE", f"High 60D correlation ({max_corr:.2f})"
        else:
            decision, reason = "PASS", "Within portfolio risk limits"

        decisions.append(decision)
        max_corrs.append(max_corr)
        reasons.append(reason)

    out["Risk Gate"] = decisions
    out["Max 60D Pair Corr"] = max_corrs
    out["Risk Gate Reason"] = reasons

    offdiag = corr.copy()
    if not offdiag.empty:
        np.fill_diagonal(offdiag.values, np.nan)
    summary = pd.DataFrame({
        "Metric": ["Model Active Positions", "Open Risk %", "Available Risk Budget %", "Highest Sector Exposure %", "Highest Pair Correlation"],
        "Value": [
            len(active),
            open_risk_pct,
            max(0, max_open_risk_pct - open_risk_pct) if not pd.isna(open_risk_pct) else np.nan,
            sector_exp.max() if len(sector_exp) else 0,
            offdiag.max().max() if not offdiag.empty else np.nan,
        ],
    })
    return out, summary


def add_position_sizing(df: pd.DataFrame, portfolio_value: float, risk_pct: float, atr_mult: float) -> pd.DataFrame:
    out = df.copy()
    risk_budget = portfolio_value * risk_pct / 100.0
    stop_distance = out["ATR14"] * atr_mult
    out["Risk Budget $"] = risk_budget
    out["Suggested Stop"] = out["Price"] - stop_distance
    out["Suggested Shares"] = np.floor(risk_budget / stop_distance.replace(0, np.nan))
    out["Suggested Position $"] = out["Suggested Shares"] * out["Price"]
    out["Position %"] = out["Suggested Position $"] / portfolio_value * 100
    return out


def actionable_setups(df: pd.DataFrame, min_sector: float, min_stock: float, min_conf: float, min_rvol: float) -> pd.DataFrame:
    near = df["Distance to Breakout %"].between(-5, 5, inclusive="both")
    mask = (
        (df["Sector Momentum Score"] >= min_sector) & (df["Stock Momentum Score"] >= min_stock)
        & (df["Signal Confidence"] >= min_conf) & (df["Price"] > df["SMA 50"])
        & (df["SMA 50"] > df["SMA 200"]) & (df["MACD"] > df["MACD Signal"])
        & near & (df["RVOL"].fillna(0) >= min_rvol)
    )
    out = df[mask].copy()
    out["Setup Status"] = np.select(
        [out["20D Breakout"] & (out["RVOL"] >= 1.5), out["Distance to Breakout %"].between(-2.5, 0), out["Distance to Breakout %"].between(-5, -2.5)],
        ["BREAKOUT", "NEAR BREAKOUT", "WATCH"], default="OTHER",
    )
    return out.sort_values(["Composite Leadership Score", "Signal Confidence"], ascending=False)


def factor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Selected Return %", "Vol Adj Momentum", "RS SPY 1M %", "RS SPY 3M %",
        "RS QQQ 1M %", "Distance 50D %", "Weak Day Strength %", "RVOL", "Persistence 21D",
    ]
    return df[cols].corr()


def attribution_table(row: pd.Series) -> pd.DataFrame:
    rows = []
    for col in row.index:
        if str(col).startswith("Contrib::"):
            rows.append({"Factor": str(col).replace("Contrib::", ""), "Contribution": safe_float(row[col])})
    return pd.DataFrame(rows).sort_values("Contribution", ascending=False) if rows else pd.DataFrame()


def build_backtest_score_matrix(close_df: pd.DataFrame, benchmark: pd.Series, tickers: List[str]) -> pd.DataFrame:
    tickers = [t for t in tickers if t in close_df.columns]
    c = close_df[tickers]
    r21 = c.pct_change(21, fill_method=None)
    r63 = c.pct_change(63, fill_method=None)
    rs21 = r21.sub(benchmark.pct_change(21, fill_method=None), axis=0)
    rs63 = r63.sub(benchmark.pct_change(63, fill_method=None), axis=0)
    d50 = c / c.rolling(50).mean() - 1
    va = r63 / (c.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)).replace(0, np.nan)
    scores = pd.DataFrame(index=c.index, columns=tickers, dtype=float)
    for dt in c.index:
        f = pd.DataFrame({"R21": r21.loc[dt], "R63": r63.loc[dt], "RS21": rs21.loc[dt], "RS63": rs63.loc[dt], "D50": d50.loc[dt], "VA": va.loc[dt]})
        raw = 0.20*zscore(f["R21"]) + 0.20*zscore(f["R63"]) + 0.20*zscore(f["RS21"]) + 0.15*zscore(f["RS63"]) + 0.10*zscore(f["D50"]) + 0.15*zscore(f["VA"])
        scores.loc[dt] = pct_rank(raw)
    return scores


def walk_forward_backtest(
    score_matrix: pd.DataFrame,
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    benchmark: pd.Series,
    universe: pd.DataFrame,
    threshold: float,
    top_n: int,
    hold_days: int,
    slippage_bps: float,
    commission_bps: float,
) -> pd.DataFrame:
    dates = score_matrix.index
    trades = []
    for i in range(252, len(dates) - hold_days - 2, 5):
        signal_date, entry_date, exit_date = dates[i], dates[i+1], dates[i+1+hold_days]
        picks = score_matrix.loc[signal_date].dropna()
        picks = picks[picks >= threshold].sort_values(ascending=False).head(top_n)
        for ticker, score in picks.items():
            if ticker not in open_df.columns or ticker not in close_df.columns:
                continue
            entry = safe_float(open_df.loc[entry_date, ticker]) if entry_date in open_df.index else np.nan
            exitp = safe_float(close_df.loc[exit_date, ticker]) if exit_date in close_df.index else np.nan
            if pd.isna(entry) or pd.isna(exitp) or entry <= 0:
                continue
            gross = (exitp / entry - 1) * 100
            costs = (slippage_bps + commission_bps) * 2 / 100.0
            net = gross - costs
            b0 = safe_float(benchmark.loc[entry_date]) if entry_date in benchmark.index else np.nan
            b1 = safe_float(benchmark.loc[exit_date]) if exit_date in benchmark.index else np.nan
            br = (b1 / b0 - 1) * 100 if not pd.isna(b0) and not pd.isna(b1) and b0 != 0 else np.nan
            sector = universe.loc[universe["Ticker"] == ticker, "Sector"]
            trades.append({
                "Ticker": ticker, "Sector": sector.iloc[0] if len(sector) else "Unknown",
                "Signal Date": signal_date, "Entry Date": entry_date, "Exit Date": exit_date,
                "Entry Price": entry, "Exit Price": exitp, "Signal Score": score,
                "Gross Return %": gross, "Costs %": costs, "Net Return %": net,
                "Benchmark Return %": br, "Excess Return %": net - br if not pd.isna(br) else np.nan,
                "Win": net > 0,
            })
    return pd.DataFrame(trades)


def backtest_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"Trades": 0, "Win Rate %": np.nan, "Avg Net Return %": np.nan, "Avg Excess Return %": np.nan, "Profit Factor": np.nan}
    wins = trades.loc[trades["Net Return %"] > 0, "Net Return %"].sum()
    losses = -trades.loc[trades["Net Return %"] < 0, "Net Return %"].sum()
    return {
        "Trades": len(trades), "Win Rate %": trades["Win"].mean() * 100,
        "Avg Net Return %": trades["Net Return %"].mean(), "Avg Excess Return %": trades["Excess Return %"].mean(),
        "Profit Factor": wins / losses if losses > 0 else np.nan,
    }


def oos_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    x = trades.sort_values("Signal Date").reset_index(drop=True)
    n = len(x)
    parts = {
        "TRAIN": x.iloc[: int(n*0.60)],
        "VALIDATION": x.iloc[int(n*0.60): int(n*0.80)],
        "TEST": x.iloc[int(n*0.80):],
    }
    return pd.DataFrame([{"Period": name, **backtest_stats(part)} for name, part in parts.items()])


def stock_history_pro(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = stock_history(raw, ticker)
    if out.empty:
        return out
    if "Volume" in out.columns:
        out["RVOL20"] = out["Volume"] / out["Volume"].shift(1).rolling(20).mean()
    return out


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=600, template="plotly_dark", paper_bgcolor="#050b11", plot_bgcolor="#050b11")
    return fig


# =============================================================================
# PRO UI
# =============================================================================

st.sidebar.markdown("## Sector Momentum Pro")
universe_name = st.sidebar.selectbox("Universe", ["S&P 500 + NASDAQ-100", "S&P 500", "NASDAQ-100"])
benchmark_choice = st.sidebar.selectbox("Benchmark", ["SPY", "QQQ"])
lookback_label = st.sidebar.selectbox("Chart / Analysis Window", list(LOOKBACKS.keys()))
ranking_horizon = st.sidebar.selectbox("Stock Ranking Horizon", ["1M", "3M", "6M"], index=1)

st.sidebar.markdown("### Signal Filters")
minimum_sector_score = st.sidebar.slider("Minimum Sector Score", 0, 100, 70, 5)
minimum_stock_score = st.sidebar.slider("Minimum Stock Score", 0, 100, 80, 5)
minimum_confidence = st.sidebar.slider("Minimum Confidence", 0, 100, 65, 5)
minimum_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 3.0, 0.8, 0.1)

st.sidebar.markdown("### Risk Model")
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1000.0, value=100000.0, step=5000.0)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.10, 2.00, 0.50, 0.05)
atr_multiple = st.sidebar.slider("ATR Stop Multiple", 1.0, 4.0, 2.0, 0.25)
max_sector_exposure = st.sidebar.slider("Max Sector Exposure (%)", 10, 50, 30, 5)

st.sidebar.markdown("### Backtest Assumptions")
slippage_bps = st.sidebar.slider("Slippage (bps)", 0, 25, 5, 1)
commission_bps = st.sidebar.slider("Commission (bps)", 0, 10, 0, 1)

if st.sidebar.button("Refresh Market Data", use_container_width=True):
    st.cache_data.clear()

st.markdown('<div class="terminal-title">Sector Momentum Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="terminal-subtitle">Regime-aware sector rotation, stock momentum, volume confirmation, persistence, risk sizing, walk-forward testing, and diagnostics.</div>',
    unsafe_allow_html=True,
)

display_days, download_period = LOOKBACKS[lookback_label]

try:
    universe = build_universe()
except Exception as exc:
    st.error(f"Constituent universe failed to load: {exc}")
    st.stop()

if universe_name == "S&P 500":
    universe = universe[universe["SP500"]].copy()
elif universe_name == "NASDAQ-100":
    universe = universe[universe["NASDAQ100"]].copy()

universe = universe[universe["Sector"] != "Unknown"].drop_duplicates("Ticker").reset_index(drop=True)
tickers = sorted(set(universe["Ticker"].tolist() + ["SPY", "QQQ", "^VIX"]))

with st.spinner("Loading market data, validating it, and building the model..."):
    raw = download_prices(tuple(tickers), download_period)

if raw.empty:
    st.error("No market data was returned.")
    st.stop()

open_df, high_df, low_df = field(raw, "Open"), field(raw, "High"), field(raw, "Low")
close_df, volume_df = field(raw, "Close"), field(raw, "Volume")

if "SPY" not in close_df.columns or "QQQ" not in close_df.columns:
    st.error("SPY or QQQ benchmark data is unavailable.")
    st.stop()

spy, qqq = close_df["SPY"].dropna(), close_df["QQQ"].dropna()
benchmark = spy if benchmark_choice == "SPY" else qqq
vix = close_df["^VIX"].dropna() if "^VIX" in close_df.columns else None

quality = data_quality_report(universe, close_df)
valid_tickers = quality.loc[quality["Status"] == "OK", "Ticker"].tolist()
universe_valid = universe[universe["Ticker"].isin(valid_tickers)].reset_index(drop=True)

breadth50, breadth200 = market_breadth_pro(close_df, valid_tickers)
regime, exposure, regime_score, vix_value = market_regime_pro(spy, qqq, breadth50, breadth200, vix)

sector_idx = build_sector_indices(close_df, universe_valid)
sector_metrics = sector_scores(sector_idx, benchmark, universe_valid, close_df, volume_df)
sector_metrics = add_sector_rotation(sector_metrics, sector_score_history(sector_idx, benchmark, 30))

persistence_hist = simplified_stock_history(close_df, benchmark, valid_tickers, 30)
persistence = persistence_features(persistence_hist)

stocks = stock_scores_pro(
    universe_valid, close_df, high_df, low_df, volume_df, spy, qqq,
    sector_metrics, persistence, ranking_horizon, regime,
)
stocks = add_signal_trajectory(stocks, persistence_hist)
stocks = add_breakout_quality(stocks)
stocks = update_signal_state_machine(stocks)
stocks = add_position_sizing(stocks, portfolio_value, risk_per_trade, atr_multiple)
setups = actionable_setups(stocks, minimum_sector_score, minimum_stock_score, minimum_confidence, minimum_rvol)
risk_source = setups if not setups.empty else stocks.head(40)
risk_source, portfolio_risk_summary = portfolio_risk_gate(
    risk_source, close_df, portfolio_value, max_sector_exposure
)
if not setups.empty:
    setups = risk_source

if sector_metrics.empty or stocks.empty:
    st.error("Momentum calculations returned no usable results.")
    st.stop()

top_sector, top_stock = sector_metrics.iloc[0], stocks.iloc[0]

k = st.columns(7)
k[0].metric("Market Regime", regime, f"{exposure:.0%} exposure multiplier")
k[1].metric("Breadth >50D", f"{breadth50:.1f}%")
k[2].metric("Breadth >200D", f"{breadth200:.1f}%")
k[3].metric("VIX", f"{vix_value:.1f}" if not pd.isna(vix_value) else "N/A")
k[4].metric("Leading Sector", top_sector["Sector"], f"{top_sector['Momentum Score']:.1f}")
k[5].metric("Top Stock", top_stock["Ticker"], f"{top_stock['Composite Leadership Score']:.1f}")
k[6].metric("Actionable Setups", len(setups), f"{len(stocks)} scored")

(
    overview_tab, rotation_tab, setups_tab, screener_tab,
    analysis_tab, signal_risk_tab, backtest_tab, diagnostics_tab, quality_tab,
) = st.tabs([
    "Overview", "Sector Rotation", "Actionable Setups", "Stock Screener",
    "Stock Analysis", "Signal & Portfolio Risk", "Backtest & Robustness", "Model Diagnostics", "Data Quality",
])

with overview_tab:
    st.subheader("Sector Momentum Heatmap")
    st.plotly_chart(sector_treemap(sector_metrics), use_container_width=True)
    left, right = st.columns([0.50, 0.50])
    with left:
        st.subheader("Sector Leaderboard")
        cols = [
            "Rank", "Sector", "Momentum Score", "Leadership", "Rotation Status",
            "Score Change 5D", "Rank Change 5D", "RS 1M %", "RS 3M %", "RS 6M %",
            "Breadth >21D %", "Breadth >50D %", "Breadth >200D %", "20D High Breadth %", "Median RVOL",
        ]
        st.dataframe(sector_metrics[cols].round(2), use_container_width=True, hide_index=True, height=520)
    with right:
        st.subheader("Top Composite Leaders")
        cols = [
            "Composite Rank", "Ticker", "Sector", "Composite Leadership Score",
            "Stock Momentum Score", "Signal Confidence", "Persistence Badge",
            "Volume Confirmation", "Trend Signal",
        ]
        st.dataframe(stocks[cols].head(30).round(2), use_container_width=True, hide_index=True, height=520)

with rotation_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fastest Improving")
        x = sector_metrics.sort_values("Score Change 5D", ascending=False).head(5)
        st.dataframe(x[["Sector", "Momentum Score", "Score Change 5D", "Score Change 10D", "Rank Change 5D", "Rotation Status"]].round(2), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Fastest Deteriorating")
        x = sector_metrics.sort_values("Score Change 5D", ascending=True).head(5)
        st.dataframe(x[["Sector", "Momentum Score", "Score Change 5D", "Score Change 10D", "Rank Change 5D", "Rotation Status"]].round(2), use_container_width=True, hide_index=True)
    st.caption("Rotation tracks changes in cross-sectional sector relative-strength scores over recent sessions.")

with setups_tab:
    st.subheader("Actionable Setups")
    cols = [
        "Ticker", "Company", "Sector", "Setup Status", "State", "Momentum Trajectory", "Signal Age", "Breakout Quality", "ATR Extension", "Risk Gate", "Risk Gate Reason", "Trend Signal",
        "Composite Leadership Score", "Stock Momentum Score", "Sector Momentum Score", "Signal Confidence",
        "Distance to Breakout %", "RVOL", "Avg RVOL 5D", "Volume Confirmation", "RSI 14", "ATR %",
        "Suggested Stop", "Suggested Shares", "Suggested Position $", "Position %",
    ]
    st.dataframe(setups[cols].round(2), use_container_width=True, hide_index=True, height=650)
    st.download_button("Export Actionable Setups", setups.to_csv(index=False).encode("utf-8"), "actionable_setups.csv", "text/csv")
    st.info("Risk sizing is a research tool based on ATR and your sidebar assumptions; it is not personalized investment advice.")

with screener_tab:
    st.subheader("Full Stock Momentum Screener")
    sectors = ["All"] + sorted(stocks["Sector"].unique().tolist())
    selected_sector = st.selectbox("Sector Filter", sectors)
    x = stocks.copy()
    if selected_sector != "All":
        x = x[x["Sector"] == selected_sector]
    x = x[
        (x["Sector Momentum Score"] >= minimum_sector_score)
        & (x["Stock Momentum Score"] >= minimum_stock_score)
        & (x["Signal Confidence"] >= minimum_confidence)
        & (x["RVOL"].fillna(0) >= minimum_rvol)
    ]
    cols = [
        "Composite Rank", "Ticker", "Company", "Sector", "Price", "Selected Return %",
        "1M Return %", "3M Return %", "RS SPY 1M %", "Vol Adj Momentum", "Realized Vol 20D %",
        "RSI 14", "RVOL", "Avg RVOL 5D", "Up/Down Volume Ratio", "Volume Confirmation",
        "Persistence 21D", "Days Top 10%", "Persistence Badge", "Distance 50D %", "Distance 200D %",
        "MACD State", "20D Breakout", "Trend Signal", "Signal Confidence",
        "Stock Momentum Score", "Sector Momentum Score", "Composite Leadership Score",
    ]
    st.dataframe(x[cols].round(2), use_container_width=True, hide_index=True, height=760)
    st.download_button("Export Screener", x.to_csv(index=False).encode("utf-8"), "sector_momentum_screener.csv", "text/csv")

with analysis_tab:
    labels = {r["Ticker"]: f"{r['Ticker']} — {r['Company']} — {r['Sector']}" for _, r in stocks.iterrows()}
    ticker = st.selectbox("Stock", stocks["Ticker"].tolist(), format_func=lambda t: labels.get(t, t))
    row = stocks[stocks["Ticker"] == ticker].iloc[0]
    m = st.columns(8)
    m[0].metric("Price", f"${row['Price']:.2f}")
    m[1].metric("Stock Score", f"{row['Stock Momentum Score']:.1f}")
    m[2].metric("Sector Score", f"{row['Sector Momentum Score']:.1f}")
    m[3].metric("Confidence", f"{row['Signal Confidence']:.0f}")
    m[4].metric("RSI", f"{row['RSI 14']:.1f}")
    m[5].metric("RVOL", f"{row['RVOL']:.2f}x")
    m[6].metric("ATR %", f"{row['ATR %']:.2f}%")
    m[7].metric("Signal", row["Trend Signal"])

    hist = stock_history_pro(raw, ticker)
    if not hist.empty:
        st.plotly_chart(stock_chart(hist, ticker, display_days), use_container_width=True)

    st.subheader("Why This Ranks")
    attr = attribution_table(row)
    a, b = st.columns([0.45, 0.55])
    with a:
        if not attr.empty:
            fig = px.bar(attr, x="Contribution", y="Factor", orientation="h", title="Factor Contributions")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#050b11", plot_bgcolor="#050b11", height=430)
            st.plotly_chart(fig, use_container_width=True)
    with b:
        snap = pd.DataFrame({
            "Metric": [
                "Persistence Badge", "Persistence 21D", "Days Top 10%", "Volume Confirmation",
                "Avg RVOL 5D", "Up/Down Volume Ratio", "Weak-Day Strength", "Distance to 20D Breakout",
                "Distance to 52W High", "Suggested Stop", "Suggested Shares", "Suggested Position $", "Position %",
            ],
            "Value": [
                row["Persistence Badge"], f"{row['Persistence 21D']:.1f}", f"{row['Days Top 10%']:.0f}", row["Volume Confirmation"],
                f"{row['Avg RVOL 5D']:.2f}x", f"{row['Up/Down Volume Ratio']:.2f}", f"{row['Weak Day Strength %']:+.3f}%",
                f"{row['Distance to Breakout %']:+.2f}%", f"{row['Distance 52W High %']:+.2f}%",
                f"${row['Suggested Stop']:.2f}", f"{row['Suggested Shares']:.0f}", f"${row['Suggested Position $']:,.0f}", f"{row['Position %']:.2f}%",
            ],
        })
        st.dataframe(snap, use_container_width=True, hide_index=True)


with signal_risk_tab:
    st.subheader("Signal Lifecycle & Portfolio Risk")
    counts = stocks["State"].fillna("UNKNOWN").value_counts().rename_axis("State").reset_index(name="Count")
    c1, c2 = st.columns([0.45, 0.55])
    with c1:
        fig = px.bar(counts, x="State", y="Count", title="WATCH → ARMED → CONFIRMED → ACTIVE → COOLDOWN")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#050b11", plot_bgcolor="#050b11", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(portfolio_risk_summary.round(2), use_container_width=True, hide_index=True)

    lifecycle = risk_source if not risk_source.empty else stocks
    cols = [
        "Ticker", "Sector", "State", "State Reason", "Momentum Trajectory", "Signal Age",
        "Peak Historical Score", "Score Decay", "Score Change 10D", "Breakout Quality",
        "ATR Extension", "Stock Momentum Score", "Signal Confidence", "Risk Gate",
        "Max 60D Pair Corr", "Risk Gate Reason",
    ]
    cols = [c for c in cols if c in lifecycle.columns]
    st.dataframe(lifecycle[cols].round(2), use_container_width=True, hide_index=True, height=620)
    st.caption("Correlation uses the latest 60 daily returns. Portfolio risk controls combine ATR open risk, sector concentration, and pairwise correlation.")

with backtest_tab:
    st.subheader("Walk-Forward Backtest")
    threshold = st.slider("Backtest Score Threshold", 70, 95, 90, 5)
    hold_days = st.selectbox("Holding Period", [10, 21, 42], index=1)
    top_n = st.slider("Top N Stocks per Rebalance", 5, 25, 10, 5)

    with st.spinner("Building historical score matrix..."):
        score_matrix = build_backtest_score_matrix(close_df, benchmark, valid_tickers)
    trades = walk_forward_backtest(
        score_matrix, close_df, open_df, benchmark, universe_valid,
        threshold, top_n, hold_days, slippage_bps, commission_bps,
    )
    stats = backtest_stats(trades)
    c = st.columns(5)
    c[0].metric("Trades", stats["Trades"])
    c[1].metric("Win Rate", f"{stats['Win Rate %']:.1f}%" if not pd.isna(stats["Win Rate %"]) else "N/A")
    c[2].metric("Avg Net Return", f"{stats['Avg Net Return %']:.2f}%" if not pd.isna(stats["Avg Net Return %"]) else "N/A")
    c[3].metric("Avg Excess Return", f"{stats['Avg Excess Return %']:.2f}%" if not pd.isna(stats["Avg Excess Return %"]) else "N/A")
    c[4].metric("Profit Factor", f"{stats['Profit Factor']:.2f}" if not pd.isna(stats["Profit Factor"]) else "N/A")

    st.subheader("Chronological 60/20/20 Validation")
    st.dataframe(oos_summary(trades).round(2), use_container_width=True, hide_index=True)
    st.subheader("Trade Audit")
    st.dataframe(trades.round(2), use_container_width=True, hide_index=True, height=520)
    st.caption("Signals are calculated on the signal date and executed at the next session's open. Today's index membership is used, so point-in-time membership is still required for institutional-grade historical research.")

with diagnostics_tab:
    st.subheader("Factor Correlation")
    corr = factor_correlation(stocks)
    st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
    pairs = []
    for i, a in enumerate(corr.columns):
        for j, b in enumerate(corr.columns):
            if j > i and abs(corr.loc[a, b]) >= 0.80:
                pairs.append({"Factor A": a, "Factor B": b, "Correlation": corr.loc[a, b]})
    st.subheader("Potentially Redundant Factors (|corr| ≥ 0.80)")
    st.dataframe(pd.DataFrame(pairs).round(2), use_container_width=True, hide_index=True)
    st.subheader("Current Regime Weights")
    st.dataframe(pd.DataFrame([{"Factor": k, "Weight": v} for k, v in regime_weights(regime).items()]), use_container_width=True, hide_index=True)

with quality_tab:
    st.subheader("Market Data Quality Gate")
    c = st.columns(5)
    c[0].metric("Universe", len(universe))
    c[1].metric("Passed", int((quality["Status"] == "OK").sum()))
    c[2].metric("Excluded", int((quality["Status"] != "OK").sum()))
    c[3].metric("Outlier Flags", int(quality["Outlier Flag"].sum()))
    c[4].metric("Latest Market Date", f"{close_df.index.max():%Y-%m-%d}")
    st.dataframe(quality.sort_values(["Status", "Ticker"]), use_container_width=True, hide_index=True, height=700)
    st.download_button("Export Data Quality Report", quality.to_csv(index=False).encode("utf-8"), "data_quality_report.csv", "text/csv")

st.markdown("---")
st.caption(
    f"{APP_NAME} • Data through {close_df.index.max():%Y-%m-%d} • "
    f"{len(universe_valid)} securities passed the data-quality gate • Regime: {regime} • "
    f"Regime score: {regime_score} • Benchmark: {benchmark_choice} • "
    f"Max sector exposure setting: {max_sector_exposure}% • Research and educational use only."
)
