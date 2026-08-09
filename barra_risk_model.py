# %% [markdown]
# # Institutional Multi-Factor Equity Risk Model (Barra-Style)
#
# **A fundamental factor risk model built from first principles**, in the style of
# MSCI Barra / Axioma commercial risk models.
#
# The model:
#
# 1. Defines a liquid US equity estimation universe with GICS-style sector classifications
# 2. Constructs point-in-time **style factor exposures** (Beta, Momentum, Volatility, Size, Reversal, Liquidity)
# 3. Estimates daily **factor returns** via cross-sectional weighted least squares (WLS)
# 4. Builds the **factor covariance matrix** (EWMA + Newey-West adjustment) and the
#    **specific (idiosyncratic) risk model**
# 5. Assembles the full asset covariance matrix `Σ = B F Bᵀ + D`
# 6. Decomposes portfolio risk into factor vs. specific components and computes
#    **ex-ante tracking error** for a sample momentum-tilted portfolio
# 7. Validates the model out-of-sample with **bias statistics**
#    (predicted vs. realized volatility)
#
# ---
# *All exposures are lagged one day relative to the returns they explain — the model is
# strictly point-in-time and free of look-ahead bias.*

# %% [markdown]
# ## Cell 1 — Environment Setup & Imports

# %%
"""Environment setup.

In Google Colab, numpy/pandas/matplotlib/seaborn are pre-installed.
yfinance is installed on demand. If market data cannot be downloaded
(offline session, rate-limited API), the pipeline falls back to a
synthetic market generator so the full workflow always runs end-to-end.
"""

from __future__ import annotations

import sys
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Optional dependency: yfinance (installed on demand in Colab) -------------
try:
    import yfinance as yf

    YF_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "yfinance"]
        )
        import yfinance as yf

        YF_AVAILABLE = True
    except Exception:
        YF_AVAILABLE = False
        print("[WARN] yfinance unavailable — the pipeline will use synthetic data.")

# --- Global plotting style (consistent, publication-quality) ------------------
PALETTE = {
    "primary": "#1f4e79",
    "accent": "#c0392b",
    "secondary": "#2e86ab",
    "neutral": "#7f8c8d",
    "positive": "#1e8449",
    "negative": "#922b21",
}

sns.set_theme(
    style="whitegrid",
    rc={
        "figure.figsize": (12, 6),
        "figure.dpi": 110,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.alpha": 0.35,
        "font.family": "sans-serif",
    },
)

print(f"numpy {np.__version__} | pandas {pd.__version__} | yfinance: {YF_AVAILABLE}")

# %% [markdown]
# ## Cell 2 — Model Configuration
#
# All tunable parameters live in a single frozen dataclass so the model is
# reproducible and easy to sweep. Half-lives follow industry convention:
# ~90 days for factor covariance, ~60 days for specific risk.

# %%
@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for the risk model pipeline."""

    # --- Data window ---
    start_date: str = "2019-01-01"
    end_date: Optional[str] = None          # None => today

    # --- Style exposure windows (trading days) ---
    beta_window: int = 252                  # rolling market beta
    momentum_lookback: int = 252            # 12-month momentum ...
    momentum_gap: int = 21                  # ... skipping the most recent month (12-1)
    vol_window: int = 63                    # 3-month realized volatility
    reversal_window: int = 21               # 1-month short-term reversal
    size_window: int = 63                   # ADV window for the size proxy
    liquidity_short: int = 21               # short ADV window (liquidity trend)
    liquidity_long: int = 126               # long ADV window

    # --- Cross-sectional regression ---
    winsor_z: float = 3.0                   # winsorization bound for z-scores
    weight_clip: tuple = (0.2, 5.0)         # relative clip on WLS regression weights
    min_names_per_date: int = 40            # skip dates with too few valid names

    # --- Covariance estimation ---
    factor_cov_halflife: float = 90.0       # EWMA half-life for factor covariance
    specific_halflife: float = 60.0         # EWMA half-life for specific risk
    newey_west_lags: int = 2                # Bartlett-weighted autocovariance lags
    specific_shrinkage: float = 0.25        # shrink specific vol toward XS mean

    # --- Validation ---
    validation_step: int = 21               # monthly bias-statistic checkpoints
    min_cov_obs: int = 150                  # min factor-return obs before predicting

    # --- General ---
    annualization: int = 252
    max_missing_frac: float = 0.10          # drop tickers missing >10% of prices
    random_seed: int = 42


CFG = ModelConfig()
print(CFG)

# %% [markdown]
# ## Cell 3 — Estimation Universe & Sector Classification
#
# ~94 large-cap, long-listed US equities across 11 GICS-style sectors.
# In production, this universe would come from a point-in-time index
# constituent file (to avoid survivorship bias) — flagged as an upgrade below.

# %%
SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "AAPL": "InfoTech", "MSFT": "InfoTech", "NVDA": "InfoTech", "AVGO": "InfoTech",
    "ORCL": "InfoTech", "CRM": "InfoTech", "ADBE": "InfoTech", "AMD": "InfoTech",
    "INTC": "InfoTech", "TXN": "InfoTech", "QCOM": "InfoTech", "ACN": "InfoTech",
    # Health Care
    "LLY": "HealthCare", "UNH": "HealthCare", "JNJ": "HealthCare", "ABBV": "HealthCare",
    "MRK": "HealthCare", "TMO": "HealthCare", "ABT": "HealthCare", "PFE": "HealthCare",
    "AMGN": "HealthCare", "GILD": "HealthCare",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "SCHW": "Financials", "BLK": "Financials", "AXP": "Financials",
    "C": "Financials", "USB": "Financials",
    # Consumer Discretionary
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "MCD": "ConsDisc",
    "NKE": "ConsDisc", "SBUX": "ConsDisc", "LOW": "ConsDisc", "TJX": "ConsDisc",
    "BKNG": "ConsDisc",
    # Consumer Staples
    "PG": "ConsStaples", "KO": "ConsStaples", "PEP": "ConsStaples", "COST": "ConsStaples",
    "WMT": "ConsStaples", "MDLZ": "ConsStaples", "CL": "ConsStaples", "KMB": "ConsStaples",
    "GIS": "ConsStaples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "PSX": "Energy", "MPC": "Energy", "OXY": "Energy",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "GE": "Industrials", "DE": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
    "UPS": "Industrials", "MMM": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "AEP": "Utilities", "EXC": "Utilities",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "FCX": "Materials",
    "NEM": "Materials", "NUE": "Materials",
    # Communication Services
    "GOOGL": "CommServices", "META": "CommServices", "NFLX": "CommServices",
    "DIS": "CommServices", "CMCSA": "CommServices", "T": "CommServices",
    "VZ": "CommServices", "TMUS": "CommServices",
    # Real Estate
    "PLD": "RealEstate", "AMT": "RealEstate", "EQIX": "RealEstate",
    "SPG": "RealEstate", "O": "RealEstate", "PSA": "RealEstate",
}

UNIVERSE = sorted(SECTOR_MAP.keys())
SECTORS = sorted(set(SECTOR_MAP.values()))
print(f"Universe: {len(UNIVERSE)} names across {len(SECTORS)} sectors")
print(pd.Series(SECTOR_MAP).value_counts().to_string())

# %% [markdown]
# ## Cell 4 — Data Collection Pipeline
#
# Downloads adjusted close prices and volumes with retry logic, then validates
# coverage. If the download fails or coverage is too thin, a **synthetic market
# generator** produces a realistic factor-structured panel so the rest of the
# notebook always executes.

# %%
class DataError(RuntimeError):
    """Raised when market data cannot be retrieved or fails validation."""


def download_market_data(
    tickers: list[str], cfg: ModelConfig, max_retries: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted prices and volumes from Yahoo Finance.

    Returns
    -------
    (prices, volumes) : tuple of DataFrame
        Wide panels indexed by date with one column per ticker.

    Raises
    ------
    DataError
        If yfinance is unavailable, the download repeatedly fails, or
        the returned panel is empty / too sparse.
    """
    if not YF_AVAILABLE:
        raise DataError("yfinance is not installed in this environment.")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=cfg.start_date,
                end=cfg.end_date,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
            if raw is None or len(raw) == 0:
                raise DataError("Empty response from Yahoo Finance.")

            prices = raw["Close"].copy()
            volumes = raw["Volume"].copy()

            # Single-ticker downloads return flat columns — normalize shape.
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(tickers[0])
                volumes = volumes.to_frame(tickers[0])

            coverage = prices.notna().mean().mean()
            if coverage < 0.5:
                raise DataError(f"Panel too sparse (coverage={coverage:.1%}).")

            print(
                f"[OK] Downloaded {prices.shape[1]} tickers × "
                f"{prices.shape[0]} days (coverage {coverage:.1%})"
            )
            return prices, volumes

        except Exception as exc:  # noqa: BLE001 — retry on any transient failure
            last_err = exc
            wait = 2 ** attempt
            print(f"[WARN] Download attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(wait)

    raise DataError(f"All download attempts failed. Last error: {last_err}")


def generate_synthetic_market(
    tickers: list[str],
    sector_map: dict[str, str],
    cfg: ModelConfig,
    n_days: int = 1500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic equity panel with genuine factor structure.

    Returns follow a linear factor model:
        r_it = beta_i * m_t + sector_{s(i),t} + mom_load_i * mom_t + eps_it
    so the cross-sectional regression engine has real structure to recover.
    This is a *fallback* for offline execution — results with real data will
    differ, but every downstream computation is identical.
    """
    rng = np.random.default_rng(cfg.random_seed)
    sectors = sorted(set(sector_map.values()))
    n, k = len(tickers), len(sectors)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    n_days = len(dates)  # bdate_range can trim when `end` falls on a weekend

    # --- True (latent) loadings ---
    beta = rng.normal(1.0, 0.30, n)                       # market beta dispersion
    mom_load = rng.normal(0.0, 1.0, n)                    # momentum-style loading
    sector_idx = np.array([sectors.index(sector_map[t]) for t in tickers])

    # --- Factor return series ---
    mkt = rng.normal(4e-4, 0.010, n_days)                 # market: ~10% ann drift
    sect = rng.normal(0.0, 0.005, (n_days, k))            # sector factors
    mom_f = rng.normal(1e-4, 0.003, n_days)               # style factor
    idio_vol = rng.uniform(0.008, 0.022, n)               # heterogeneous idio vol
    eps = rng.standard_normal((n_days, n)) * idio_vol

    rets = (
        np.outer(mkt, beta)
        + sect[:, sector_idx]
        + np.outer(mom_f, mom_load) * 0.5
        + eps
    )
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=tickers
    )

    # --- Volumes: persistent size dispersion + noise ---
    base_vol = np.exp(rng.uniform(13.0, 17.5, n))         # spans ~2 orders of magnitude
    vol_noise = np.exp(rng.normal(0.0, 0.30, (n_days, n)))
    volumes = pd.DataFrame(base_vol * vol_noise, index=dates, columns=tickers)

    print(f"[OK] Synthetic market generated: {n} tickers × {n_days} days")
    return prices, volumes


# --- Execute: real data with synthetic fallback -------------------------------
try:
    prices_raw, volumes_raw = download_market_data(UNIVERSE, CFG)
    DATA_SOURCE = "Yahoo Finance"
except DataError as e:
    print(f"[FALLBACK] {e}")
    prices_raw, volumes_raw = generate_synthetic_market(UNIVERSE, SECTOR_MAP, CFG)
    DATA_SOURCE = "Synthetic"

print(f"Data source: {DATA_SOURCE}")

# %% [markdown]
# ## Cell 5 — Data Cleaning & Return Construction
#
# * Drop tickers with excessive missing data (delistings, bad symbols)
# * Forward-fill short gaps only (≤ 5 days) — never back-fill (look-ahead)
# * Compute simple daily returns, clip data-error outliers at ±50%
# * Build the equal-weighted **market return** used for beta estimation

# %%
def clean_market_data(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    sector_map: dict[str, str],
    cfg: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict[str, str]]:
    """Validate, clean, and align the raw price/volume panels.

    Returns
    -------
    prices, volumes, returns : DataFrame
        Cleaned wide panels on a common date index / ticker set.
    market_ret : Series
        Equal-weighted universe return (beta benchmark).
    sector_map_clean : dict
        Sector map restricted to surviving tickers.
    """
    # 1) Drop tickers with too much missing history --------------------------
    missing_frac = prices.isna().mean()
    keep = missing_frac[missing_frac <= cfg.max_missing_frac].index.tolist()
    dropped = sorted(set(prices.columns) - set(keep))
    if dropped:
        print(f"[CLEAN] Dropping {len(dropped)} sparse tickers: {dropped}")
    prices = prices[keep].copy()
    volumes = volumes[keep].copy()

    # 2) Fill only short gaps (holidays / single missing prints) -------------
    prices = prices.ffill(limit=5)
    volumes = volumes.ffill(limit=5)

    # 3) Restrict to dates where most of the universe trades -----------------
    valid_rows = prices.notna().mean(axis=1) >= 0.90
    prices, volumes = prices.loc[valid_rows], volumes.loc[valid_rows]

    # 4) Returns, with data-error clipping -----------------------------------
    returns = prices.pct_change()
    n_clipped = int((returns.abs() > 0.50).sum().sum())
    if n_clipped:
        print(f"[CLEAN] Clipping {n_clipped} extreme return observations (>|50%|)")
    returns = returns.clip(-0.50, 0.50)
    returns = returns.iloc[1:]  # drop the first NaN row

    # 5) Equal-weighted market proxy -----------------------------------------
    market_ret = returns.mean(axis=1).rename("MKT_EW")

    sector_map_clean = {t: sector_map[t] for t in prices.columns}

    # Sanity checks -----------------------------------------------------------
    assert prices.index.is_monotonic_increasing, "Dates must be sorted."
    assert not returns.empty, "Return panel is empty after cleaning."
    sector_counts = pd.Series(sector_map_clean).value_counts()
    thin = sector_counts[sector_counts < 3]
    if len(thin):
        print(f"[WARN] Thin sectors (<3 names): {thin.to_dict()}")

    print(
        f"[OK] Clean panel: {returns.shape[1]} tickers × {returns.shape[0]} days "
        f"({returns.index[0].date()} → {returns.index[-1].date()})"
    )
    return prices, volumes, returns, market_ret, sector_map_clean


prices, volumes, returns, market_ret, sector_map = clean_market_data(
    prices_raw, volumes_raw, SECTOR_MAP, CFG
)
TICKERS = list(returns.columns)
SECTORS = sorted(set(sector_map.values()))

# %% [markdown]
# ## Cell 6 — Style Factor Exposures
#
# Six style descriptors, all computed **strictly from past data**:
#
# | Factor | Definition | Captures |
# |---|---|---|
# | Beta | 252d rolling beta vs. EW universe | Systematic market sensitivity |
# | Momentum | 12-month return, skipping last month (12-1) | Medium-term trend premium |
# | Volatility | 63d realized vol (annualized) | Low-vol anomaly / risk appetite |
# | Size | log(63d average dollar volume) | Large vs. small (ADV proxy for mcap) |
# | Reversal | 21d return | Short-term mean reversion |
# | Liquidity | log(ADV 21d / ADV 126d) | Trading-activity trend |
#
# Each descriptor is **cross-sectionally z-scored per date**, winsorized at ±3σ,
# and re-standardized — the standard Barra treatment.

# %%
def cross_sectional_standardize(df: pd.DataFrame, winsor_z: float) -> pd.DataFrame:
    """Z-score each row (date) across the universe, winsorize, re-standardize.

    Degenerate rows (zero cross-sectional dispersion) are returned as NaN
    rather than dividing by zero.
    """
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    z = z.clip(-winsor_z, winsor_z)
    # Re-standardize so winsorized exposures are exactly mean-0 / unit-vol
    mu2 = z.mean(axis=1)
    sd2 = z.std(axis=1).replace(0.0, np.nan)
    return z.sub(mu2, axis=0).div(sd2, axis=0)


def compute_style_exposures(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    market_ret: pd.Series,
    cfg: ModelConfig,
) -> dict[str, pd.DataFrame]:
    """Compute the raw style descriptor panels, then standardize each.

    Returns a dict {factor_name: DataFrame(date × ticker)} of z-scored exposures.
    """
    ann = np.sqrt(cfg.annualization)
    dollar_vol = (prices * volumes).replace(0.0, np.nan)

    raw: dict[str, pd.DataFrame] = {}

    # --- Beta: rolling cov(r_i, r_m) / var(r_m) ------------------------------
    cov_im = returns.rolling(cfg.beta_window, min_periods=cfg.beta_window // 2).cov(
        market_ret
    )
    var_m = market_ret.rolling(
        cfg.beta_window, min_periods=cfg.beta_window // 2
    ).var()
    raw["Beta"] = cov_im.div(var_m, axis=0)

    # --- Momentum: 12-1 month price return -----------------------------------
    raw["Momentum"] = (
        prices.shift(cfg.momentum_gap) / prices.shift(cfg.momentum_lookback) - 1.0
    )

    # --- Volatility: 63d realized, annualized --------------------------------
    raw["Volatility"] = (
        returns.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std() * ann
    )

    # --- Size: log average dollar volume (point-in-time mcap proxy) ----------
    raw["Size"] = np.log(
        dollar_vol.rolling(cfg.size_window, min_periods=cfg.size_window // 2).mean()
    )

    # --- Short-term reversal: trailing 21d return ----------------------------
    raw["Reversal"] = prices / prices.shift(cfg.reversal_window) - 1.0

    # --- Liquidity: trend in trading activity --------------------------------
    adv_s = dollar_vol.rolling(cfg.liquidity_short, min_periods=10).mean()
    adv_l = dollar_vol.rolling(cfg.liquidity_long, min_periods=60).mean()
    raw["Liquidity"] = np.log(adv_s / adv_l)

    return {
        name: cross_sectional_standardize(panel, cfg.winsor_z)
        for name, panel in raw.items()
    }


style_exposures = compute_style_exposures(prices, volumes, returns, market_ret, CFG)
STYLE_FACTORS = list(style_exposures.keys())

# Quick QC: latest-date exposure summary (should be ~N(0,1) cross-sectionally)
qc = pd.DataFrame(
    {name: panel.iloc[-1] for name, panel in style_exposures.items()}
).describe().T[["mean", "std", "min", "max"]]
print("Latest-date exposure QC (expect mean≈0, std≈1):")
print(qc.round(3).to_string())

# %% [markdown]
# ## Cell 7 — Cross-Sectional Regression Engine
#
# For each date *t* we estimate the fundamental factor model
#
# $$ r_{i,t} \;=\; \sum_{s} X^{ind}_{i,s}\, f^{ind}_{s,t} \;+\; \sum_{k} X^{sty}_{i,k,\,t-1}\, f^{sty}_{k,t} \;+\; \varepsilon_{i,t} $$
#
# via **WLS** with √(dollar-volume) regression weights (large, liquid names get
# more weight — the Barra convention uses √mcap; ADV is our point-in-time proxy).
#
# * Industry dummies partition the universe, so they collectively absorb the
#   market intercept (no separate constant is needed).
# * Style exposures are mean-zero per date, so the two blocks are well-conditioned.
# * **Exposures are lagged one day** relative to the returns — no look-ahead.

# %%
def build_regression_weights(
    prices: pd.DataFrame, volumes: pd.DataFrame, cfg: ModelConfig
) -> pd.DataFrame:
    """√(average dollar volume) WLS weights, normalized and clipped per date."""
    adv = (prices * volumes).rolling(cfg.size_window, min_periods=20).mean()
    w = np.sqrt(adv)
    w = w.div(w.mean(axis=1), axis=0)                      # mean 1 per date
    return w.clip(cfg.weight_clip[0], cfg.weight_clip[1])  # bound influence


def run_cross_sectional_regressions(
    returns: pd.DataFrame,
    style_exposures: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    reg_weights: pd.DataFrame,
    cfg: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Estimate daily factor returns by cross-sectional WLS.

    Returns
    -------
    factor_returns : DataFrame (date × factor)
        Daily estimated returns to industry and style factors.
    residuals : DataFrame (date × ticker)
        Specific (idiosyncratic) returns ε_it.
    r_squared : Series
        Weighted cross-sectional R² per date — the share of daily return
        dispersion explained by the factor structure.
    """
    tickers = list(returns.columns)
    sectors = sorted(set(sector_map.values()))
    styles = list(style_exposures.keys())
    factor_names = sectors + styles
    n_factors = len(factor_names)

    # Lag all exposures and weights by one day (point-in-time discipline)
    lagged_styles = {k: v.shift(1) for k, v in style_exposures.items()}
    lagged_w = reg_weights.shift(1)

    # Static industry dummy matrix (ticker × sector)
    ind_dummies = pd.get_dummies(pd.Series(sector_map)).reindex(
        index=tickers, columns=sectors, fill_value=0
    ).astype(float)

    fr_rows, resid_rows, r2_rows, kept_dates = [], [], [], []
    skipped = 0

    for t in returns.index:
        y_all = returns.loc[t]

        # Assemble the style exposure block for this date
        style_block = pd.DataFrame(
            {k: lagged_styles[k].loc[t] for k in styles}, index=tickers
        )
        w_all = lagged_w.loc[t]

        # Valid names: finite return, all exposures, and a weight
        valid = (
            y_all.notna()
            & style_block.notna().all(axis=1)
            & w_all.notna()
        )
        n_valid = int(valid.sum())
        if n_valid < max(cfg.min_names_per_date, n_factors + 10):
            skipped += 1
            continue

        idx = valid[valid].index
        y = y_all.loc[idx].to_numpy()
        X = np.hstack(
            [ind_dummies.loc[idx].to_numpy(), style_block.loc[idx].to_numpy()]
        )
        w = w_all.loc[idx].to_numpy()

        # Drop industry columns with no members today (keeps X full rank)
        col_ok = X[:, : len(sectors)].sum(axis=0) > 0
        col_mask = np.concatenate([col_ok, np.ones(len(styles), dtype=bool)])
        Xm = X[:, col_mask]

        # WLS via row-scaling: minimize Σ w_i (y_i − x_i·f)²
        sw = np.sqrt(w)
        try:
            f_hat, *_ = np.linalg.lstsq(Xm * sw[:, None], y * sw, rcond=None)
        except np.linalg.LinAlgError:
            skipped += 1
            continue

        # Scatter estimates back into the full factor vector (NaN if absent)
        f_full = np.full(n_factors, np.nan)
        f_full[col_mask] = f_hat

        resid = y - Xm @ f_hat
        ybar = np.average(y, weights=w)
        ss_res = np.sum(w * resid**2)
        ss_tot = np.sum(w * (y - ybar) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        fr_rows.append(f_full)
        resid_rows.append(pd.Series(resid, index=idx))
        r2_rows.append(r2)
        kept_dates.append(t)

    factor_returns = pd.DataFrame(fr_rows, index=kept_dates, columns=factor_names)
    residuals = pd.DataFrame(resid_rows, index=kept_dates).reindex(columns=tickers)
    r_squared = pd.Series(r2_rows, index=kept_dates, name="XS_R2")

    print(
        f"[OK] Estimated {len(kept_dates)} daily cross-sections "
        f"({skipped} skipped in burn-in / sparse dates); "
        f"mean weighted R² = {r_squared.mean():.1%}"
    )
    return factor_returns, residuals, r_squared


reg_weights = build_regression_weights(prices, volumes, CFG)
factor_returns, residuals, r_squared = run_cross_sectional_regressions(
    returns, style_exposures, sector_map, reg_weights, CFG
)
FACTOR_NAMES = list(factor_returns.columns)

# %% [markdown]
# ## Cell 8 — Factor Return Analytics
#
# Before trusting the covariance model, inspect the estimated factor returns:
# annualized premia, volatilities, Sharpe ratios, and Newey-West-free t-stats
# of the mean. Style factor t-stats near ±2 indicate an economically meaningful
# premium over the sample.

# %%
def factor_summary_table(fr: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Annualized performance and significance summary per factor."""
    fr = fr.dropna(how="all")
    n_obs = fr.notna().sum()
    mean_d, std_d = fr.mean(), fr.std()
    out = pd.DataFrame(
        {
            "AnnRet_%": mean_d * cfg.annualization * 100,
            "AnnVol_%": std_d * np.sqrt(cfg.annualization) * 100,
            "Sharpe": (mean_d / std_d) * np.sqrt(cfg.annualization),
            "t_stat": mean_d / (std_d / np.sqrt(n_obs)),
            "N_obs": n_obs,
        }
    )
    return out.round(2)


summary = factor_summary_table(factor_returns, CFG)
print("=== Style factor summary ===")
print(summary.loc[STYLE_FACTORS].to_string())
print("\n=== Industry factor summary ===")
print(summary.loc[SECTORS].to_string())

# %% [markdown]
# ## Cell 9 — Factor Covariance Matrix (EWMA + Newey-West)
#
# The factor covariance matrix `F` is estimated with exponentially weighted
# moving-average (EWMA) weights (half-life 90d) plus a Bartlett-weighted
# **Newey-West** adjustment for serial correlation in daily factor returns.
# Eigenvalues are clipped to guarantee the matrix is positive semi-definite —
# a hard requirement for any optimizer consuming this model.

# %%
def nearest_psd(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone via eigenvalue clipping."""
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    clipped = int((vals < eps).sum())
    if clipped:
        vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T


def ewma_covariance(
    x: pd.DataFrame,
    halflife: float,
    nw_lags: int = 0,
    annualization: int = 252,
) -> pd.DataFrame:
    """EWMA covariance with optional Newey-West serial-correlation adjustment.

    Parameters
    ----------
    x : DataFrame (T × K)
        Factor return history (rows with any NaN are dropped).
    halflife : float
        EWMA half-life in observations.
    nw_lags : int
        Number of Bartlett-weighted autocovariance lags to include.
    annualization : int
        Periods per year for scaling.
    """
    xv = x.dropna(how="any")
    T, K = xv.shape
    if T < 2 * K:
        warnings.warn(
            f"Only {T} observations for a {K}-factor covariance — "
            "estimates will be noisy."
        )

    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()

    xd = xv.to_numpy() - w @ xv.to_numpy()          # EWMA-demeaned
    cov = (xd * w[:, None]).T @ xd                  # Γ₀ (weighted)

    for lag in range(1, nw_lags + 1):
        bartlett = 1.0 - lag / (nw_lags + 1.0)
        g = (xd[lag:] * w[lag:, None]).T @ xd[:-lag]
        cov += bartlett * (g + g.T)

    cov = nearest_psd(cov) * annualization
    return pd.DataFrame(cov, index=x.columns, columns=x.columns)


factor_cov = ewma_covariance(
    factor_returns,
    halflife=CFG.factor_cov_halflife,
    nw_lags=CFG.newey_west_lags,
    annualization=CFG.annualization,
)

# Factor correlation matrix for diagnostics / plotting
_d = np.sqrt(np.diag(factor_cov.to_numpy()))
factor_corr = pd.DataFrame(
    factor_cov.to_numpy() / np.outer(_d, _d),
    index=FACTOR_NAMES,
    columns=FACTOR_NAMES,
)

eigvals = np.linalg.eigvalsh(factor_cov.to_numpy())
print(
    f"[OK] Factor covariance: {factor_cov.shape[0]} factors | "
    f"min eigenvalue = {eigvals.min():.2e} (PSD ✓) | "
    f"condition number = {eigvals.max() / eigvals.min():.1e}"
)

# %% [markdown]
# ## Cell 10 — Specific (Idiosyncratic) Risk Model
#
# Specific risk is the EWMA volatility of each stock's regression residuals,
# **shrunk toward the cross-sectional mean** to stabilize thin estimates —
# the same reason Barra applies Bayesian shrinkage to specific risk.

# %%
def estimate_specific_risk(
    residuals: pd.DataFrame, cfg: ModelConfig, as_of: Optional[pd.Timestamp] = None
) -> pd.Series:
    """Annualized specific volatility per stock at a given date.

    σᵢ = (1 − s)·σ̂ᵢ + s·mean(σ̂)  with shrinkage intensity s.
    """
    hist = residuals if as_of is None else residuals.loc[:as_of]
    ew_std = hist.ewm(halflife=cfg.specific_halflife, min_periods=40).std()
    latest = ew_std.iloc[-1] * np.sqrt(cfg.annualization)

    xs_mean = latest.mean()
    shrunk = (1.0 - cfg.specific_shrinkage) * latest + cfg.specific_shrinkage * xs_mean
    return shrunk.rename("specific_vol")


specific_vol = estimate_specific_risk(residuals, CFG)
print("Specific risk distribution (annualized):")
print(specific_vol.describe().round(3).to_string())

# %% [markdown]
# ## Cell 11 — The Assembled Risk Model:  Σ = B F Bᵀ + D
#
# `RiskModel` bundles the exposure matrix, factor covariance, and specific
# variances at an analysis date, and provides:
#
# * total / factor / specific portfolio volatility
# * per-factor **risk contributions** (Euler decomposition)
# * **ex-ante tracking error** vs. a benchmark, with active-exposure attribution

# %%
class RiskModel:
    """Barra-style asset covariance model at a single analysis date."""

    def __init__(
        self,
        exposures: pd.DataFrame,        # ticker × factor
        factor_cov: pd.DataFrame,       # factor × factor (annualized)
        specific_vol: pd.Series,        # ticker (annualized vol)
    ) -> None:
        # Align everything on a common, fully-observed ticker set
        common = (
            exposures.dropna().index.intersection(specific_vol.dropna().index)
        )
        if len(common) == 0:
            raise ValueError("No tickers with complete exposures and specific risk.")

        self.tickers = list(common)
        self.factors = list(factor_cov.columns)
        self.B = exposures.loc[common, self.factors].to_numpy(dtype=float)
        self.F = factor_cov.to_numpy(dtype=float)
        self.spec_var = (specific_vol.loc[common].to_numpy(dtype=float)) ** 2

    # ---------------------------------------------------------------- utils --
    def _check_weights(self, w: pd.Series) -> np.ndarray:
        wv = w.reindex(self.tickers).fillna(0.0).to_numpy(dtype=float)
        if np.all(wv == 0):
            raise ValueError("Weight vector has no overlap with the model universe.")
        return wv

    # ------------------------------------------------------------- interface --
    def portfolio_variance(self, w: pd.Series) -> dict[str, float]:
        """Total, factor, and specific variance (annualized) of a portfolio."""
        wv = self._check_weights(w)
        x = self.B.T @ wv                            # portfolio factor exposures
        var_factor = float(x @ self.F @ x)
        var_specific = float(wv**2 @ self.spec_var)
        return {
            "total_var": var_factor + var_specific,
            "factor_var": var_factor,
            "specific_var": var_specific,
        }

    def portfolio_vol(self, w: pd.Series) -> float:
        """Annualized total volatility."""
        return float(np.sqrt(self.portfolio_variance(w)["total_var"]))

    def risk_decomposition(self, w: pd.Series) -> pd.DataFrame:
        """Euler decomposition: each factor's contribution to total variance.

        Contribution_k = x_k · (F x)_k, which sums to the factor variance;
        expressed both in variance terms and as a share of total risk.
        """
        wv = self._check_weights(w)
        x = self.B.T @ wv
        contrib = x * (self.F @ x)
        var = self.portfolio_variance(w)

        rows = pd.DataFrame(
            {"exposure": x, "var_contribution": contrib}, index=self.factors
        )
        rows.loc["— Specific —"] = [np.nan, var["specific_var"]]
        rows["pct_of_total_var"] = rows["var_contribution"] / var["total_var"] * 100
        rows["vol_contribution_%"] = (
            rows["var_contribution"] / np.sqrt(var["total_var"]) * 100
        )
        return rows.sort_values("var_contribution", ascending=False)

    def tracking_error(self, w: pd.Series, w_bench: pd.Series) -> dict:
        """Ex-ante annualized tracking error and its decomposition."""
        active = (
            w.reindex(self.tickers).fillna(0.0)
            - w_bench.reindex(self.tickers).fillna(0.0)
        )
        var = self.portfolio_variance(active)
        return {
            "tracking_error": float(np.sqrt(var["total_var"])),
            "factor_te": float(np.sqrt(var["factor_var"])),
            "specific_te": float(np.sqrt(var["specific_var"])),
            "active_weights": active,
        }


def exposures_at(
    date: pd.Timestamp,
    style_exposures: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    tickers: list[str],
    sectors: list[str],
) -> pd.DataFrame:
    """Assemble the full (ticker × factor) exposure matrix as of a date."""
    ind = pd.get_dummies(pd.Series(sector_map)).reindex(
        index=tickers, columns=sectors, fill_value=0
    ).astype(float)
    sty = pd.DataFrame(
        {k: v.loc[:date].iloc[-1] for k, v in style_exposures.items()},
        index=tickers,
    )
    return pd.concat([ind, sty], axis=1)


# --- Build the model at the latest date --------------------------------------
AS_OF = returns.index[-1]
B_now = exposures_at(AS_OF, style_exposures, sector_map, TICKERS, SECTORS)
model = RiskModel(B_now, factor_cov, specific_vol)
print(f"[OK] Risk model built as of {AS_OF.date()} over {len(model.tickers)} names")

# %% [markdown]
# ## Cell 12 — Sample Portfolio: Momentum Tilt vs. Equal-Weight Benchmark
#
# A realistic use case: a PM runs a 20-name momentum-tilted long-only book and
# wants to know (a) how risky it is, (b) *where* the risk comes from, and
# (c) the ex-ante tracking error vs. their benchmark.

# %%
def build_momentum_portfolio(
    model: RiskModel, style_exposures: dict, as_of: pd.Timestamp, n_names: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Top-N momentum names, equal-weighted, vs. EW universe benchmark."""
    mom = style_exposures["Momentum"].loc[:as_of].iloc[-1].reindex(model.tickers)
    top = mom.nlargest(n_names).index
    w_port = pd.Series(1.0 / n_names, index=top, name="momentum_tilt")
    w_bench = pd.Series(
        1.0 / len(model.tickers), index=model.tickers, name="benchmark_ew"
    )
    return w_port, w_bench


w_port, w_bench = build_momentum_portfolio(model, style_exposures, AS_OF)

port_var = model.portfolio_variance(w_port)
bench_vol = model.portfolio_vol(w_bench)
decomp = model.risk_decomposition(w_port)
te = model.tracking_error(w_port, w_bench)

print(f"=== Momentum-tilt portfolio ({len(w_port)} names) — as of {AS_OF.date()} ===")
print(f"Predicted total vol   : {np.sqrt(port_var['total_var']):7.2%}")
print(f"  · factor vol        : {np.sqrt(port_var['factor_var']):7.2%}")
print(f"  · specific vol      : {np.sqrt(port_var['specific_var']):7.2%}")
print(f"Benchmark (EW) vol    : {bench_vol:7.2%}")
print(f"Ex-ante tracking error: {te['tracking_error']:7.2%}  "
      f"(factor {te['factor_te']:.2%} / specific {te['specific_te']:.2%})")
print("\nTop risk contributors (% of total variance):")
print(decomp.head(8).round(3).to_string())

# %% [markdown]
# ## Cell 13 — Out-of-Sample Model Validation: Bias Statistics
#
# The industry-standard test of a risk model: at each month-end, predict next
# month's portfolio volatility, then compare with what actually happened.
#
# Standardized returns  $z_t = r_{t+1} / \hat\sigma_t$  should have
# **std ≈ 1** ("bias statistic"). Values > 1 mean the model under-predicts
# risk; < 1 means it over-predicts. Everything below uses only information
# available at each prediction date.

# %%
def run_bias_test(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    residuals: pd.DataFrame,
    style_exposures: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    cfg: ModelConfig,
) -> pd.DataFrame:
    """Walk-forward monthly volatility forecasts for the EW universe portfolio.

    Returns a DataFrame with predicted vol, realized vol, and standardized
    return z per validation window.
    """
    tickers = list(returns.columns)
    sectors = sorted(set(sector_map.values()))
    fr_clean = factor_returns.dropna(how="any")
    step = cfg.validation_step

    records = []
    dates = returns.index
    # Start once enough factor-return history exists to estimate F
    first_ok = fr_clean.index[min(cfg.min_cov_obs, len(fr_clean) - 1)]
    start_pos = dates.searchsorted(first_ok)

    for pos in range(start_pos, len(dates) - step, step):
        t = dates[pos]

        # --- Point-in-time model components --------------------------------
        fr_hist = fr_clean.loc[:t]
        if len(fr_hist) < cfg.min_cov_obs:
            continue
        F_t = ewma_covariance(
            fr_hist, cfg.factor_cov_halflife, cfg.newey_west_lags, cfg.annualization
        )
        spec_t = estimate_specific_risk(residuals, cfg, as_of=t)
        B_t = exposures_at(t, style_exposures, sector_map, tickers, sectors)

        try:
            m_t = RiskModel(B_t, F_t, spec_t)
        except ValueError:
            continue

        w_ew = pd.Series(1.0 / len(m_t.tickers), index=m_t.tickers)
        sigma_ann = m_t.portfolio_vol(w_ew)

        # --- Realized outcome over the next `step` days ---------------------
        fwd = returns.iloc[pos + 1 : pos + 1 + step][m_t.tickers].mean(axis=1)
        realized_ann = fwd.std() * np.sqrt(cfg.annualization)
        r_period = float((1.0 + fwd).prod() - 1.0)
        sigma_period = sigma_ann * np.sqrt(step / cfg.annualization)

        records.append(
            {
                "date": t,
                "predicted_vol": sigma_ann,
                "realized_vol": realized_ann,
                "z": r_period / sigma_period if sigma_period > 0 else np.nan,
            }
        )

    out = pd.DataFrame(records).set_index("date")
    if out.empty:
        raise RuntimeError("Bias test produced no observations — check data window.")

    bias = out["z"].std()
    corr = out[["predicted_vol", "realized_vol"]].corr().iloc[0, 1]
    print(
        f"[OK] Bias test: {len(out)} monthly forecasts | "
        f"bias statistic = {bias:.3f} (target ≈ 1.0) | "
        f"pred-vs-realized vol correlation = {corr:.2f}"
    )
    return out


bias_df = run_bias_test(
    returns, factor_returns, residuals, style_exposures, sector_map, CFG
)
BIAS_STAT = bias_df["z"].std()

# %% [markdown]
# ## Cell 14 — Visualization Suite
#
# Eight production-quality charts: factor performance, correlation structure,
# model fit, exposures, risk decomposition, specific risk, and validation.

# %%
def fmt_pct(ax, axis: str = "y") -> None:
    """Format an axis as percentages."""
    formatter = mtick.PercentFormatter(xmax=1.0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


# ---- Fig 1: Cumulative style factor returns ---------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
cum_style = (1.0 + factor_returns[STYLE_FACTORS].fillna(0.0)).cumprod() - 1.0
for col in STYLE_FACTORS:
    ax.plot(cum_style.index, cum_style[col], lw=1.6, label=col)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Cumulative Style Factor Returns (Cross-Sectional WLS Estimates)")
ax.set_ylabel("Cumulative return")
fmt_pct(ax)
ax.legend(ncol=3, frameon=False)
plt.tight_layout()
plt.savefig("fig1_style_factor_returns.png", bbox_inches="tight")
plt.show()

# ---- Fig 2: Factor correlation heatmap --------------------------------------
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(factor_corr, dtype=bool), k=1)
sns.heatmap(
    factor_corr,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    annot=False,
    linewidths=0.4,
    cbar_kws={"label": "Correlation", "shrink": 0.8},
    ax=ax,
)
ax.set_title("Factor Return Correlation Matrix (EWMA, Newey-West Adjusted)")
plt.tight_layout()
plt.savefig("fig2_factor_correlation.png", bbox_inches="tight")
plt.show()

# ---- Fig 3: Cross-sectional R² ----------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(
    r_squared.index, r_squared, color=PALETTE["neutral"], lw=0.6, alpha=0.5,
    label="Daily",
)
roll = r_squared.rolling(63).mean()
ax.plot(roll.index, roll, color=PALETTE["primary"], lw=2.2, label="63d average")
ax.set_title("Cross-Sectional R²: Share of Daily Return Dispersion Explained")
ax.set_ylabel("Weighted R²")
fmt_pct(ax)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig3_cross_sectional_r2.png", bbox_inches="tight")
plt.show()

# ---- Fig 4: Style exposure heatmap (portfolio holdings) ---------------------
fig, ax = plt.subplots(figsize=(9, 8))
holdings_expo = B_now.loc[w_port.index, STYLE_FACTORS]
sns.heatmap(
    holdings_expo,
    cmap="RdBu_r",
    center=0,
    vmin=-3,
    vmax=3,
    annot=True,
    fmt=".1f",
    annot_kws={"size": 7},
    linewidths=0.4,
    cbar_kws={"label": "Exposure (z-score)", "shrink": 0.8},
    ax=ax,
)
ax.set_title(f"Style Exposures — Momentum-Tilt Holdings (as of {AS_OF.date()})")
plt.tight_layout()
plt.savefig("fig4_exposure_heatmap.png", bbox_inches="tight")
plt.show()

# ---- Fig 5: Portfolio risk decomposition ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[2, 1])

top_contrib = decomp["pct_of_total_var"].head(10)[::-1]
colors = [
    PALETTE["accent"] if f == "— Specific —" else PALETTE["primary"]
    for f in top_contrib.index
]
axes[0].barh(top_contrib.index, top_contrib.values, color=colors, alpha=0.9)
axes[0].set_title("Top Contributors to Portfolio Variance")
axes[0].set_xlabel("% of total variance")
axes[0].axvline(0, color="black", lw=0.8)

fac_share = port_var["factor_var"] / port_var["total_var"]
axes[1].pie(
    [fac_share, 1 - fac_share],
    labels=["Factor", "Specific"],
    colors=[PALETTE["primary"], PALETTE["accent"]],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
axes[1].set_title("Factor vs. Specific Variance")
fig.suptitle(
    f"Risk Decomposition — Momentum Tilt "
    f"(σ = {np.sqrt(port_var['total_var']):.1%}, "
    f"TE vs EW = {te['tracking_error']:.1%})",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("fig5_risk_decomposition.png", bbox_inches="tight")
plt.show()

# ---- Fig 6: Specific risk landscape -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(
    specific_vol.dropna(), bins=25, color=PALETTE["secondary"],
    edgecolor="white", alpha=0.9,
)
axes[0].axvline(
    specific_vol.mean(), color=PALETTE["accent"], ls="--", lw=1.5,
    label=f"Mean = {specific_vol.mean():.1%}",
)
axes[0].set_title("Specific Risk Distribution (Annualized)")
axes[0].set_xlabel("Specific volatility")
fmt_pct(axes[0], "x")
axes[0].legend(frameon=False)

sector_spec = (
    specific_vol.groupby(pd.Series(sector_map)).mean().sort_values()
)
axes[1].barh(sector_spec.index, sector_spec.values, color=PALETTE["primary"], alpha=0.9)
axes[1].set_title("Average Specific Risk by Sector")
axes[1].set_xlabel("Specific volatility")
fmt_pct(axes[1], "x")
plt.tight_layout()
plt.savefig("fig6_specific_risk.png", bbox_inches="tight")
plt.show()

# ---- Fig 7: Predicted vs. realized volatility -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].plot(
    bias_df.index, bias_df["predicted_vol"], color=PALETTE["primary"], lw=2,
    label="Predicted (ex-ante)",
)
axes[0].plot(
    bias_df.index, bias_df["realized_vol"], color=PALETTE["accent"], lw=1.4,
    alpha=0.85, label="Realized (next month)",
)
axes[0].set_title("Predicted vs. Realized Volatility — EW Universe Portfolio")
axes[0].set_ylabel("Annualized volatility")
fmt_pct(axes[0])
axes[0].legend(frameon=False)

lims = [
    0.0,
    max(bias_df["predicted_vol"].max(), bias_df["realized_vol"].max()) * 1.1,
]
axes[1].scatter(
    bias_df["predicted_vol"], bias_df["realized_vol"],
    color=PALETTE["primary"], alpha=0.7, s=35, edgecolor="white",
)
axes[1].plot(lims, lims, color=PALETTE["neutral"], ls="--", lw=1.2, label="45° line")
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel("Predicted vol")
axes[1].set_ylabel("Realized vol")
fmt_pct(axes[1])
fmt_pct(axes[1], "x")
axes[1].set_title(f"Forecast Accuracy (Bias Statistic = {BIAS_STAT:.2f})")
axes[1].legend(frameon=False)
plt.tight_layout()
plt.savefig("fig7_bias_test.png", bbox_inches="tight")
plt.show()

# ---- Fig 8: Rolling bias statistic ------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4.5))
roll_bias = bias_df["z"].rolling(12).std()
ax.plot(roll_bias.index, roll_bias, color=PALETTE["primary"], lw=2)
ax.axhline(1.0, color=PALETTE["positive"], ls="--", lw=1.4, label="Perfect calibration")
ax.fill_between(
    roll_bias.index, 0.8, 1.2, color=PALETTE["positive"], alpha=0.10,
    label="±20% band",
)
ax.set_title("Rolling 12-Month Bias Statistic (std of standardized returns)")
ax.set_ylabel("Bias statistic")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig8_rolling_bias.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Cell 15 — Model Summary Report

# %%
def print_model_report() -> None:
    """Console summary of the full risk model build and validation."""
    line = "=" * 68
    print(line)
    print("  BARRA-STYLE MULTI-FACTOR EQUITY RISK MODEL — SUMMARY REPORT")
    print(line)
    print(f"  Data source          : {DATA_SOURCE}")
    print(f"  Sample               : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Universe             : {len(TICKERS)} names / {len(SECTORS)} industries")
    print(f"  Factors              : {len(FACTOR_NAMES)} "
          f"({len(SECTORS)} industry + {len(STYLE_FACTORS)} style)")
    print(f"  Daily cross-sections : {len(factor_returns)}")
    print(f"  Mean weighted R²     : {r_squared.mean():.1%}")
    print("-" * 68)
    print(f"  Momentum-tilt portfolio (as of {AS_OF.date()}):")
    print(f"    Total vol          : {np.sqrt(port_var['total_var']):.2%}")
    print(f"    Factor / Specific  : "
          f"{port_var['factor_var'] / port_var['total_var']:.0%} / "
          f"{port_var['specific_var'] / port_var['total_var']:.0%}")
    print(f"    Tracking error     : {te['tracking_error']:.2%}")
    print("-" * 68)
    print(f"  Validation ({len(bias_df)} monthly forecasts):")
    print(f"    Bias statistic     : {BIAS_STAT:.3f}   (target ≈ 1.00)")
    corr = bias_df[["predicted_vol", "realized_vol"]].corr().iloc[0, 1]
    print(f"    Pred/realized corr : {corr:.2f}")
    verdict = (
        "well calibrated" if 0.8 <= BIAS_STAT <= 1.2
        else ("under-predicts risk" if BIAS_STAT > 1.2 else "over-predicts risk")
    )
    print(f"    Verdict            : model {verdict}")
    print(line)


print_model_report()

# %% [markdown]
# ## Cell 16 — Extensions & Production Roadmap
#
# * **Point-in-time universe** — replace the static ticker list with historical
#   index constituents to remove survivorship bias.
# * **True fundamentals** — pull book value / earnings from SEC EDGAR XBRL
#   (properly lagged) to add Value, Quality, and Earnings-Yield factors.
# * **Constrained regression** — add a Country/Market factor with the Barra
#   constraint that cap-weighted industry factor returns sum to zero.
# * **Optimizer integration** — feed `Σ = B F Bᵀ + D` into a `cvxpy`
#   mean-variance optimizer with factor-exposure bounds.
# * **Volatility regime adjustment** — Barra-style VRA scaling of the factor
#   covariance so forecasts adapt faster in crises.
# * **Performance attribution** — daily Brinson / factor-based attribution of a
#   live portfolio's realized returns.
# * **Packaging** — split into `src/riskmodel/` modules with `pytest` unit tests
#   (see the repository structure in the project README).
