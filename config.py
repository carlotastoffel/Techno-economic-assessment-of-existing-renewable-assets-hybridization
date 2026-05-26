from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import numpy as np

# =============================================================================
# 0.  GENERAL SIMULATION SETTINGS
# =============================================================================

# Temporal resolution
TIMESTEP_MIN: float = 15.0          # minutes per timestep
TIMESTEP_H: float   = TIMESTEP_MIN / 60.0   # hours per timestep  (0.25)
TIMESTEPS_PER_HOUR: int = int(60 / TIMESTEP_MIN)  # 4
TIMESTEPS_PER_DAY:  int = 24 * TIMESTEPS_PER_HOUR  # 96
TIMESTEPS_PER_YEAR: int = 365 * TIMESTEPS_PER_DAY  # 35 040

# Price-scenario years to simulate (dispatch is run for each year independently)
# Update this list to match the years in your Clean Horizon forecast dataset.
PRICE_SCENARIO_YEARS: list[int] = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]


# Rolling-window optimisation horizon (dispatch is solved week by week to keep
# the MILP tractable; BESS state is carried forward between windows).
ROLLING_WINDOW_DAYS: int = 7
ROLLING_WINDOW_STEPS: int = ROLLING_WINDOW_DAYS * TIMESTEPS_PER_DAY  # 672

# =============================================================================
# 0b.  BESS CHARGING SCENARIOS
# =============================================================================
#
# Four scenarios controlling which energy sources can charge the BESS and
# how hydro is treated commercially:
#
#  Scenario A — PV-only charging, hydro always to grid (feed-in tariff active)
#               BESS charges exclusively from PV surplus.
#               Hydro is hardwired to grid — not subject to (m,c) split.
#               No grid (DA/IDC) purchases allowed.
#
#  Scenario B — PV + grid charging, hydro always to grid (feed-in tariff active)
#               Same commercial treatment of hydro as A.
#               BESS additionally buys from DA/IDC market (pure arbitrage).
#
#  Scenario C — PV + hydro charging, no grid purchases (feed-in tariff expired)
#               Hydro treated identically to PV: subject to (m,c) split,
#               merchant share (1-c) of (PV+hydro) can charge the BESS.
#               No grid (DA/IDC) purchases allowed.
#
#  Scenario D — PV + hydro + grid charging (feed-in tariff expired, full arbitrage)
#               Same as C but BESS also buys from DA/IDC market.
#
# Production mode: set RUN_ALL_SCENARIOS = True  → all 4 run per iteration.
# Debug mode:      set RUN_ALL_SCENARIOS = False → only CHARGING_SCENARIO_DEBUG.

RUN_ALL_SCENARIOS: bool = False

CHARGING_SCENARIOS: list[str] = ['A', 'B', 'C', 'D']

# =============================================================================
# 0c.  INPUT PROFILE SCENARIOS
# =============================================================================
#
ACTIVE_PV_PROFILE:    str = "PV_profile"     
ACTIVE_HYDRO_PROFILE: str = "Hydro_profile"   

# Used only when RUN_ALL_SCENARIOS = False (for quick single-scenario debugging)
CHARGING_SCENARIO_DEBUG: str = 'A'

# Convenience: which scenarios treat hydro as a merchant asset (feed-in expired)
HYDRO_MERCHANT_SCENARIOS: set[str] = {'C', 'D'}

# Convenience: which scenarios allow grid (DA/IDC) purchases to charge BESS
GRID_CHARGING_SCENARIOS: set[str] = {'B', 'D'}

def get_active_scenarios() -> list[str]:
    """Return the list of scenarios to run based on RUN_ALL_SCENARIOS flag."""
    return CHARGING_SCENARIOS if RUN_ALL_SCENARIOS else [CHARGING_SCENARIO_DEBUG]


# =============================================================================
# 1.  HYDRO PLANT (existing asset — fixed, not a decision variable)
# =============================================================================

@dataclass(frozen=True)
class HydroConfig:

    grid_injection_capacity_mw: float = 10.0  

    installed_capacity_mw: float = 11.00        


HYDRO = HydroConfig()

# =============================================================================
# 2.  PV PLANT (decision variable — sized in grid search)
# =============================================================================

@dataclass(frozen=True)
class PVConfig:
    """PV sizing grid and technical parameters."""

    # Grid-search range  (MW AC)
    size_min_mw:  float = 2.0
    size_max_mw:  float = 12.0
    size_step_mw: float = 2.0

    # DC/AC oversize ratio (typical utility-scale)
    dc_ac_ratio: float = 1.15

    # Annual degradation rate (applied year-on-year in multi-year runs)
    degradation_rate_per_year: float = 0.005   # 0.5 % / year

    # Inverter / cabling losses (applied to SAM/PVsyst profile at load time)
    system_loss: float = 0.02   # 2 %

    @property
    def sizes_mw(self) -> list[float]:
        """All candidate PV sizes [MW AC]."""
        n = round((self.size_max_mw - self.size_min_mw) / self.size_step_mw) + 1
        return [round(self.size_min_mw + i * self.size_step_mw, 2) for i in range(n)]


PV = PVConfig()

# =============================================================================
# 3.  BATTERY ENERGY STORAGE SYSTEM — BESS (decision variable — sized in grid search)
# =============================================================================

@dataclass(frozen=True)
class BESSConfig:
    """BESS sizing grid and technical parameters."""

    # --- Electrical parameters ---
    charge_efficiency:    float = 0.96    # η_ch  (one-way)
    discharge_efficiency: float = 0.96    # η_dis (one-way)

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency

    # --- State-of-charge limits ---
    min_soc_pu: float = 0.05    # 5 %  → avoids deep discharge
    max_soc_pu: float = 0.95    # 95 % → avoids overcharge

    # --- Degradation & cycling ---
    max_cycles_per_day:  float = 1.0     # Operational limit
    max_cycles_per_year: float = 548.0   # = 1.5 cycles/day × 365

    # Calendar degradation (linear, applied per simulated year)
    calendar_degradation_per_year: float = 0.02   # 2 % capacity loss / year


    # --- Power converter limit (same for charge and discharge) ---
    @property
    def power_sizes_mw(self) -> list[float]:
        """All candidate BESS power ratings [MW]."""
        n = round((self.power_max_mw - self.power_min_mw) / self.power_step_mw) + 1
        return [round(self.power_min_mw + i * self.power_step_mw, 2) for i in range(n)]

    def capacity_mwh(self, power_mw: float, duration_h: float) -> float:
        """Energy capacity for a given power + duration combination [MWh]."""
        return power_mw * duration_h


BESS = BESSConfig()


# =============================================================================
# 4.  SIZING GRID (outer loop — all (PV, BESS_power, BESS_duration) combos)
# =============================================================================

def get_sizing_combinations() -> list[dict]:
    """
    Returns all (pv_mw, bess_power_mw, bess_duration_h, bess_energy_mwh)
    combinations for the grid search.

    Total combinations = 11 PV sizes × 15 BESS power × 3 durations = 495
    """
    combos = []
    for pv_mw, bess_mw, dur_h in product(
        PV.sizes_mw,
        BESS.power_sizes_mw,
        BESS.duration_hours,
    ):
        combos.append({
            "pv_mw":           pv_mw,
            "bess_power_mw":   bess_mw,
            "bess_duration_h": dur_h,
            "bess_energy_mwh": BESS.capacity_mwh(bess_mw, dur_h),
        })
    return combos


SIZING_COMBINATIONS: list[dict] = get_sizing_combinations()   # 495 entries

# =============================================================================
# 5.  PPA CONTRACTING SCENARIOS  (middle loop)
# =============================================================================

# Pre-set contracted share scenarios (fraction of each hour's output sold via PPA)
# c = 0 → fully merchant;  c = 1 → fully contracted
PPA_CONTRACT_SHARES: list[float] = [0.00, 0.25, 0.50, 0.75, 1.00]




# =============================================================================
# 7.  PORTUGUESE / MIBEL MARKET PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class MarketConfig:
    """Rules and parameters for each electricity market."""

    # --- Day-Ahead Market (MIBEL spot) ---
    da_gate_closure_h: int = 12    # Gate closure: noon D-1 (hour of day)
    da_settlement: str = "hourly"  # Hourly products

    # --- Intraday Continuous (IDC / XBID) ---
    idc_enabled: bool = True

    # --- aFRR (Frequency Restoration Reserve) — Portugal / REN rules ---
    afrr_enabled: bool = True

    # Minimum bid size [MW] — REN requirement
    afrr_min_bid_mw: float = 1.0

    # Energy reservation requirements (minimum SoC headroom per MW of capacity bid)
    # These ensure the BESS can honour its aFRR obligations.
    afrr_up_energy_reserve_mwh_per_mw:   float = 2   # [MWh/MW] 
    afrr_down_energy_reserve_mwh_per_mw: float = 2   # [MWh/MW] 

    # FCR: NOT included (not applicable in Portuguese market context for this study)
    fcr_enabled: bool = False

    # Imbalance / balancing market: modelled implicitly via IDC (not explicit)
    balancing_enabled: bool = False


MARKET = MarketConfig()

# =============================================================================
# 8.  FINANCIAL MODEL PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class FinancialConfig:
    """Project finance parameters for the IRR / NPV model."""

    currency: str = "EUR"

    # --- Project timeline ---
    construction_years: int = 2         # Years of CAPEX spend before operations
    operation_years:    int = 18      # Project lifetime [years]  



    # --- CAPEX [EUR/unit] — placeholder values, update when data available ---
    capex_pv_eur_per_mw:     float = 503_000.0    # EUR / MW AC installed
    capex_epc_contingency:   float = 9_000         # EUR / MW for solar 
    
    capex_2hbess_eur_per_mw: float = 619_000.0    # EUR / MW installed for a 2h hours BESS
    capex_4hbess_eur_per_mw: float = 925_000.0    # EUR / MW installed for a 4h hours BESS

    line_capex_fixed: float = 855_000.0    # Fixed grid connection cost (line + substation upgrade)

    # --- OPEX [EUR / year or EUR / MWh] ---
    opex_pv_eur_per_mw_year:    float = 16_500.0   # Fixed O&M [EUR/MW/yr]
    opex_hydro_eur_per_year:    float = 550_000.0  # Existing plant w/ 11MW O&M [EUR/yr]
    
    opex_balancing_eur_per_mwh: float = 1.75       # grid variable costs
    grid_tariff_bess_charge_eur_per_mwh: float = 4.9       # grid tariff for BESS charging from grid (applies in scenarios B and D
        

    opex_2hbess_eur_per_mw_year: float = 17_500.0      # Fixed O&M [EUR/MW/yr] for a 2h BESS
    opex_4hbess_eur_per_mw_year: float = 25_000.0   # Fixed O&M [EUR/MW/yr] for a 4h BESS

    # --- Debt financing ---
    debt_share:    float = 0    # 100% equity financing (no debt) — placeholder, update when known
    
    # --- Tax & depreciation ---
    corporate_tax_rate: float = 0.21   # Portugal corporate tax (IRC) [21 %]

    # Straight-line depreciation periods
    depreciation_years_pv:   int = 15
    depreciation_years_bess: int = 10  # BESS may be replaced/augmented at year 10

    # --- Inflation ---
    inflation_rate: float = 0.018   # 2 % / year (applied to OPEX escalation)

    


FINANCIAL = FinancialConfig()

# Feed-in Tariff price for the hydro-only baseline (Scenarios A & C)
# Set to the exact value from the plant's regulatory contract.
HYDRO_FIT_EUR_MWH: float = 98.30   # EUR/MWh — placeholder, update when known

# =============================================================================
# 9.  SOLVER SETTINGS (Gurobi via Pyomo)
# =============================================================================

@dataclass(frozen=True)
class SolverConfig:
    """Gurobi solver options for the dispatch MILP."""

    solver_name: str = "gurobi"

    # Optimality gap
    mip_gap: float = 0.03

    # Time limit per solve [seconds] — safety valve for large instances
    time_limit_s: int =120

    # Gurobi method: -1 = automatic (recommended for MILP)
    method: int = -1

    # Numeric focus: 1 = balanced, 2 = more conservative
    numeric_focus: int = 1

    # Scaling: 1 = standard, 2 = aggressive
    scale_flag: int = 2

    # Presolve: 2 = aggressive
    presolve: int = 2

    # Output verbosity during solve
    output_flag: int = 1  # Set to 1 to see Gurobi log in console

    # Log file (set to "" to disable)
    log_file: str = "gurobi_dispatch.log"

    def as_dict(self) -> dict:
        """Returns options dict for direct use with Pyomo's SolverFactory."""
        return {
            "MIPGap":       self.mip_gap,
            "TimeLimit":    self.time_limit_s,
            "Method":       self.method,
            "NumericFocus": self.numeric_focus,
            "ScaleFlag":    self.scale_flag,
            "Presolve":     self.presolve,
            "OutputFlag":   self.output_flag,
            "LogFile":      self.log_file,
        }


SOLVER = SolverConfig()

# =============================================================================
# 10.  FILE PATHS
# =============================================================================

from pathlib import Path

# Project root — folder containing config.py and all scripts
ROOT_DIR = Path(__file__).parent

# Input data folders (as they exist on disk)
INPUT_DIR         = ROOT_DIR / "Inputs"
PRICES_DIR        = ROOT_DIR / "Inputs" / "Prices Central"

# Aliases used by data_loader.py — point to the actual folders
PV_PROFILES_DIR   = INPUT_DIR    
HYDRO_INFLOW_DIR  = INPUT_DIR    
MARKET_PRICES_DIR = PRICES_DIR   # DA.csv, id_revenues.csv, aFRR_*.csv live here

# Results output folder
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 11.  QUICK SANITY CHECK  (runs when you execute config.py directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("FRAMEWORK CONFIGURATION SUMMARY")
    print("=" * 65)

    print(f"\n[Temporal resolution]")
    print(f"  Timestep          : {TIMESTEP_MIN} min  ({TIMESTEP_H:.4f} h)")
    print(f"  Steps per year    : {TIMESTEPS_PER_YEAR:,}")

    print(f"\n[Hydro plant]")
    print(f"  Capacity          : {HYDRO.installed_capacity_mw} MW")
    print(f"  Annual energy     : {HYDRO.annual_energy} MWh")
    print(f"  Grid injection cap: {HYDRO.grid_injection_capacity_mw} MW")

    print(f"\n[PV sizing grid]")
    print(f"  Sizes (MW AC)     : {PV.sizes_mw}   ({len(PV.sizes_mw)} options)")

    print(f"\n[BESS sizing grid]")
    print(f"  Power sizes (MW)  : {BESS.power_sizes_mw}   ({len(BESS.power_sizes_mw)} options)")
    print(f"  Durations (h)     : {list(BESS.duration_hours)}")
    print(f"  Round-trip eff.   : {BESS.round_trip_efficiency:.2%}")

    print(f"\n[Total sizing combinations]")
    print(f"  {len(PV.sizes_mw)} PV × {len(BESS.power_sizes_mw)} BESS power × "
          f"{len(BESS.duration_hours)} durations = {len(SIZING_COMBINATIONS)} combinations")

    print(f"\n[Charging scenarios]")
    active = get_active_scenarios()
    scenario_labels = {
        "A": "PV only            (hydro to grid, no grid buy)",
        "B": "PV + Grid          (hydro to grid, grid buy allowed)",
        "C": "PV + Hydro         (hydro merchant, no grid buy)",
        "D": "PV + Hydro + Grid  (hydro merchant, grid buy allowed)",
    }
    for s in CHARGING_SCENARIOS:
        marker = "ACTIVE" if s in active else "     "
        print(f"  [{marker}] Scenario {s}: {scenario_labels[s]}")
    print(f"  Run all scenarios : {RUN_ALL_SCENARIOS}")

    print(f"\n[PPA scenarios]")
    print(f"  Contract shares   : {PPA_CONTRACT_SHARES}")

    print(f"\n[Target IRRs]")


    print(f"\n[Markets enabled]")
    print(f"  Day-ahead (MIBEL) : ✓")
    print(f"  Intraday (IDC)    : {'✓' if MARKET.idc_enabled else '✗'}")
    print(f"  aFRR              : {'✓' if MARKET.afrr_enabled else '✗'}")
    print(f"  FCR               : {'✗ (not applicable in Portugal)'}")

    print(f"\n[Price scenario years]")
    print(f"  {PRICE_SCENARIO_YEARS}")

    print(f"\n[Financial model]")
    print(f"  Project lifetime  : {FINANCIAL.operation_years} years")
    print(f"  Debt share        : {FINANCIAL.debt_share:.0%}")
    print(f"  WACC              : {FINANCIAL.wacc:.2%}")
    print(f"  Corporate tax     : {FINANCIAL.corporate_tax_rate:.0%}")
    print(f"  CAPEX PV          : €{FINANCIAL.capex_pv_eur_per_mw:,.0f} / MW")
    print(f"  CAPEX BESS 2h     : €{FINANCIAL.capex_2hbess_eur_per_mw:,.0f} / MW")
    print(f"  CAPEX BESS 4h     : €{FINANCIAL.capex_4hbess_eur_per_mw:,.0f} / MW")

    print(f"\n[Solver]")
    print(f"  Solver            : {SOLVER.solver_name}")
    print(f"  MIP gap           : {SOLVER.mip_gap:.1%}")
    print(f"  Time limit        : {SOLVER.time_limit_s} s / solve")

    print("\n" + "=" * 65)
    print(f"Config loaded successfully.  Ready to build the framework.")
    print("=" * 65)
