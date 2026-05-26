from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config import (
    ACTIVE_HYDRO_PROFILE,
    ACTIVE_PV_PROFILE,
    HYDRO,
    HYDRO_INFLOW_DIR,
    MARKET_PRICES_DIR,
    PV,
    PV_PROFILES_DIR,
    PRICE_SCENARIO_YEARS,
    TIMESTEPS_PER_YEAR,
    TIMESTEP_H,
)

logger = logging.getLogger(__name__)

# Market key -> filename stem
_MARKET_FILES: Dict[str, str] = {
    "da":                   "DA",
    "idc":                  "id_revenues",
    "afrr_up_activation":   "aFRR_discharge",
    "afrr_up_capacity":     "cap_aFRR_discharge",
    "afrr_down_activation": "aFRR_charge",
    "afrr_down_capacity":   "cap_aFRR_charge",
}


# =============================================================================
# HELPERS
# =============================================================================

def _read_csv(path: Path) -> pd.DataFrame:
    """Read a semicolon-separated, comma-decimal CSV. '-' -> 0."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, header=0, sep=';', decimal=',')
    df = df.loc[:, ~df.columns.str.contains('Unnamed', na=False)]
    df = df.replace({'-': 0, '': 0})
    df = df.replace(r'^\s*-\s*$', 0, regex=True)
    return df.apply(pd.to_numeric, errors='coerce').fillna(0.0)


def _year_column(df: pd.DataFrame, year: int, filename: str) -> np.ndarray:
    """Extract one year column from a wide DataFrame."""
    df.columns = [int(str(c).strip()) for c in df.columns]
    if year not in df.columns:
        raise KeyError(
            f"{filename}: year {year} not found. "
            f"Available: {sorted(df.columns.tolist())}"
        )
    arr = df[year].to_numpy(dtype=np.float64)
    if len(arr) < TIMESTEPS_PER_YEAR:
        raise ValueError(
            f"{filename} year {year}: "
            f"expected {TIMESTEPS_PER_YEAR} rows, got {len(arr)}."
        )
    return arr[:TIMESTEPS_PER_YEAR]


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def load_pv_profile(pv_mw: float, year: int) -> np.ndarray:
    """
    PV generation profile for a given installed capacity and scenario year.
    profile[t] = pv_mw x reference[t] x degradation_factor(year)
    Returns np.ndarray of shape (35040,) in [MW].
    """
    path = PV_PROFILES_DIR / f"{ACTIVE_PV_PROFILE}.csv"
    raw  = _read_csv(path)
    arr  = raw.iloc[:, 0].to_numpy(dtype=np.float64)
    arr  = np.clip(arr, 0.0, None)

    if len(arr) < TIMESTEPS_PER_YEAR:
        raise ValueError(
            f"{ACTIVE_PV_PROFILE}.csv: "
            f"expected {TIMESTEPS_PER_YEAR} rows, got {len(arr)}."
        )
    arr = arr[:TIMESTEPS_PER_YEAR]

    # System losses, scaling, and annual degradation
    years_elapsed = max(0, year - PRICE_SCENARIO_YEARS[0])
    arr = (arr
           * (1.0 - PV.system_loss)
           * pv_mw
           * (1.0 - PV.degradation_rate_per_year) ** years_elapsed )# MWh

    arr=arr / TIMESTEP_H

    logger.info(
        "PV [%s] %.1f MW year %d",
        ACTIVE_PV_PROFILE, pv_mw, year,
    )
    return arr


def load_hydro_profile(year: int) -> np.ndarray:
    """
    Hydro generation profile for a given scenario year.
    Returns np.ndarray of shape (35040,) in [MW], clipped to turbine limits.
    """
    path    = HYDRO_INFLOW_DIR / f"{ACTIVE_HYDRO_PROFILE}.csv"
    df      = _read_csv(path)
    profile = _year_column(df, year, f"{ACTIVE_HYDRO_PROFILE}.csv")
    profile = profile/TIMESTEP_H
    profile = np.clip(profile, HYDRO.min_output_mw, HYDRO.max_output_mw) #MW 

    logger.info(
        "Hydro [%s] year %d",
        ACTIVE_HYDRO_PROFILE, year,
    )
    return profile


def load_market_prices(year: int) -> Dict[str, np.ndarray]:
    """
    All market price arrays for one scenario year.
    Returns dict of np.ndarray of shape (35040,) each.

    Keys: 'da', 'idc', 'afrr_up_activation', 'afrr_up_capacity',
          'afrr_down_activation', 'afrr_down_capacity'
    """
    prices = {}
    for key, stem in _MARKET_FILES.items():
        path        = MARKET_PRICES_DIR / f"{stem}.csv"
        df          = _read_csv(path)
        prices[key] = _year_column(df, year, f"{stem}.csv")

    logger.info("Market prices loaded for year %d.", year)
    return prices


def load_scenario(pv_mw: float, year: int) -> Dict:
    """
    Load all dispatch inputs for one (pv_mw, year) combination.
    Returns dict with keys: 'prices', 'pv', 'hydro', 'year', 'pv_mw'.
    """
    return {
        "prices": load_market_prices(year),
        "pv":     load_pv_profile(pv_mw, year),
        "hydro":  load_hydro_profile(year),
        "year":   year,
        "pv_mw":  pv_mw,
    }
