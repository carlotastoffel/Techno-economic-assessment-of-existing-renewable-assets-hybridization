from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BESSParameters:
    """Technical parameters for the BESS."""

    capacity_mwh: float
    power_charge_mw: float
    power_discharge_mw: float

    round_trip_efficiency: float = 0.88

    min_soc: float = 0.05
    max_soc: float = 0.95
    initial_soc: float = 0.50

    ramp_rate_pct_per_min: float = 0.40

    max_cycles_per_day: float = 1.5       # 548 / 365 ≈ 1.5 — must match max_cycles_per_year
    max_cycles_per_year: float = 548.0

    lifetime_years: float = 18.0
    eol_capacity_pct: float = 0.65

    degradation_per_cycle: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._validate()
        total_cycles = self.max_cycles_per_year * self.lifetime_years
        self.degradation_per_cycle = (
            (1.0 - self.eol_capacity_pct) / total_cycles if total_cycles > 0 else 0.0
        )

    def _validate(self) -> None:
        if self.capacity_mwh < 0:
            raise ValueError("capacity_mwh must be non-negative.")
        if self.power_charge_mw < 0 or self.power_discharge_mw < 0:
            raise ValueError("Power ratings must be non-negative.")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must be in (0, 1].")
        if not 0 <= self.min_soc < self.max_soc <= 1:
            raise ValueError("SoC limits must satisfy 0 ≤ min_soc < max_soc ≤ 1.")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("initial_soc must be within [min_soc, max_soc].")
        if self.ramp_rate_pct_per_min <= 0:
            raise ValueError("ramp_rate_pct_per_min must be positive.")
        if self.lifetime_years <= 0:
            raise ValueError("lifetime_years must be positive.")
        if not 0 < self.eol_capacity_pct < 1:
            raise ValueError("eol_capacity_pct must be in (0, 1).")


class BESSModel:
    """Stateful BESS model for rolling-window dispatch."""

    def __init__(self, params: BESSParameters) -> None:
        self.params = params
        self.charge_efficiency = np.sqrt(params.round_trip_efficiency)
        self.discharge_efficiency = np.sqrt(params.round_trip_efficiency)

        self.soc: float = params.initial_soc
        self.energy_mwh: float = self.soc * params.capacity_mwh
        self.previous_power_mw: float = 0.0
        self.cumulative_cycles: float = 0.0
        self.cumulative_throughput_mwh: float = 0.0
        self.degradation_factor: float = 1.0

        self._cycle_ref_capacity_mwh: float = self.get_available_capacity()
        self._cycle_throughput_accum_mwh: float = 0.0

        self._history: Dict[str, List] = {k: [] for k in [
            "soc",
            "power_mw",
            "energy_mwh",
            "cumulative_cycles",
            "degradation_factor",
            "available_capacity_mwh",
        ]}

    def get_available_capacity(self) -> float:
        return self.params.capacity_mwh * self.degradation_factor

    def get_usable_energy(self) -> float:
        cap = self.get_available_capacity()
        return cap * (self.params.max_soc - self.params.min_soc)

    def get_dispatch_limits(self, timestep_minutes: float = 15.0) -> Dict[str, float]:
        h = timestep_minutes / 60.0
        cap = self.get_available_capacity()

        energy_to_max = max(0.0, cap * self.params.max_soc - self.energy_mwh)
        energy_avail = max(0.0, self.energy_mwh - cap * self.params.min_soc)

        max_ch = min(
            energy_to_max / (self.charge_efficiency * h) if h > 0 else 0.0,
            self.params.power_charge_mw,
        )
        max_dis = min(
            energy_avail * self.discharge_efficiency / h if h > 0 else 0.0,
            self.params.power_discharge_mw,
        )

        return {
            "available_charge_mw": float(max(0.0, max_ch)),
            "available_discharge_mw": float(max(0.0, max_dis)),
            "energy_to_max_mwh": float(energy_to_max),
            "energy_available_mwh": float(energy_avail),
            "energy_mwh": float(self.energy_mwh),
            "soc": float(self.soc),
            "available_capacity_mwh": float(cap),
            "previous_power_mw": float(self.previous_power_mw),
            "soc_min_mwh": float(cap * self.params.min_soc),
            "soc_max_mwh": float(cap * self.params.max_soc),
        }

    def get_ramp_limit_mw(self, timestep_minutes: float = 15.0) -> float:
        max_p = max(self.params.power_charge_mw, self.params.power_discharge_mw)
        return self.params.ramp_rate_pct_per_min * timestep_minutes * max_p

    def _update_degradation(self) -> None:
        self.degradation_factor = max(
            self.params.eol_capacity_pct,
            1.0 - self.params.degradation_per_cycle * self.cumulative_cycles,
        )

    def _step(self, charge_mw: float, discharge_mw: float, timestep_minutes: float = 15.0) -> None:
        h = timestep_minutes / 60.0
        cap = self.get_available_capacity()
        if cap <= 0:
            warnings.warn("BESS available capacity is zero — state not updated.")
            return

        p_ch = max(0.0, float(charge_mw))
        p_dis = max(0.0, float(discharge_mw))

        soc_new = (
            self.soc
            + self.charge_efficiency * p_ch * h / cap
            - p_dis * h / (self.discharge_efficiency * cap)
        )
        soc_new = float(np.clip(soc_new, self.params.min_soc, self.params.max_soc))

        self.soc = soc_new
        self.energy_mwh = self.soc * cap

        # Raw power×time — consistent with MILP's throughput_step = (charge+discharge)×h.
        energy_throughput = (p_ch + p_dis) * h
        self.cumulative_throughput_mwh += energy_throughput

        if self._cycle_ref_capacity_mwh <= 0:
            self._cycle_ref_capacity_mwh = cap

        # One cycle = 2×cap of raw throughput (full-nameplate definition, consistent
        # with MILP budget: max_throughput = 2×E_bess_deg×max_cycles_per_day×days).
        self._cycle_throughput_accum_mwh += energy_throughput
        while self._cycle_throughput_accum_mwh >= 2.0 * self._cycle_ref_capacity_mwh > 0:
            self.cumulative_cycles += 1.0
            self._cycle_throughput_accum_mwh -= 2.0 * self._cycle_ref_capacity_mwh
            self._cycle_ref_capacity_mwh = cap

        self._update_degradation()
        self.previous_power_mw = p_ch - p_dis

        self._history["soc"].append(self.soc)
        self._history["power_mw"].append(self.previous_power_mw)
        self._history["energy_mwh"].append(self.energy_mwh)
        self._history["cumulative_cycles"].append(self.cumulative_cycles)
        self._history["degradation_factor"].append(self.degradation_factor)
        self._history["available_capacity_mwh"].append(cap)

    def simulate_from_dispatch(
        self,
        charge_profile_mw: np.ndarray,
        discharge_profile_mw: np.ndarray,
        timestep_minutes: float = 15.0,
    ) -> None:
        if len(charge_profile_mw) != len(discharge_profile_mw):
            raise ValueError("charge_profile_mw and discharge_profile_mw must have the same length.")

        for ch, dis in zip(charge_profile_mw, discharge_profile_mw):
            ch = float(ch)
            dis = float(dis)
            if ch > 1e-6 and dis > 1e-6:
                warnings.warn(
                    f"Simultaneous charge ({ch:.4f} MW) and discharge ({dis:.4f} MW) detected; netting the two.",
                )
                if ch >= dis:
                    ch = ch - dis
                    dis = 0.0
                else:
                    dis = dis - ch
                    ch = 0.0
            self._step(ch, dis, timestep_minutes)

    def get_state_summary(self) -> Dict:
        return {
            "soc": self.soc,
            "energy_mwh": self.energy_mwh,
            "available_capacity_mwh": self.get_available_capacity(),
            "usable_energy_mwh": self.get_usable_energy(),
            "cumulative_cycles": self.cumulative_cycles,
            "cumulative_throughput_mwh": self.cumulative_throughput_mwh,
            "degradation_factor": self.degradation_factor,
            "capacity_loss_pct": (1.0 - self.degradation_factor) * 100,
            "previous_power_mw": self.previous_power_mw,
        }

    def clone_state(self) -> Dict:
        return {
            "soc": self.soc,
            "energy_mwh": self.energy_mwh,
            "previous_power_mw": self.previous_power_mw,
            "cumulative_cycles": self.cumulative_cycles,
            "cumulative_throughput_mwh": self.cumulative_throughput_mwh,
            "degradation_factor": self.degradation_factor,
            "_cycle_ref_capacity_mwh": self._cycle_ref_capacity_mwh,
            "_cycle_throughput_accum_mwh": self._cycle_throughput_accum_mwh,
        }

    def restore_state(self, state: Dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def reset(self, initial_soc: Optional[float] = None) -> None:
        soc0 = initial_soc if initial_soc is not None else self.params.initial_soc
        if not self.params.min_soc <= soc0 <= self.params.max_soc:
            raise ValueError("initial_soc outside allowed range.")
        self.soc = soc0
        self.energy_mwh = soc0 * self.params.capacity_mwh
        self.previous_power_mw = 0.0
        self.cumulative_cycles = 0.0
        self.cumulative_throughput_mwh = 0.0
        self.degradation_factor = 1.0
        self._cycle_ref_capacity_mwh = self.get_available_capacity()
        self._cycle_throughput_accum_mwh = 0.0
        self._history = {k: [] for k in self._history}


def make_bess(power_mw: float, duration_h: float, initial_soc: float = 0.50, **kwargs) -> BESSModel:
    params = BESSParameters(
        capacity_mwh=power_mw * duration_h,
        power_charge_mw=power_mw,
        power_discharge_mw=power_mw,
        initial_soc=initial_soc,
        **kwargs,
    )
    return BESSModel(params)


def calculate_cycles_from_soc_profile(soc_profile: np.ndarray) -> float:
    return float(np.abs(np.diff(soc_profile)).sum() / 2.0)


def calculate_energy_throughput(power_profile_mw: np.ndarray, timestep_hours: float) -> float:
    return float(np.abs(power_profile_mw).sum() * timestep_hours)
