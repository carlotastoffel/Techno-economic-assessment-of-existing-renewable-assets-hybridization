from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict


import numpy as np
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    RangeSet,
    Reals,
    SolverFactory,
    Var,
    maximize,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition

from revised_bess_model import BESSModel
from config import (
    GRID_CHARGING_SCENARIOS,
    HYDRO,
    HYDRO_MERCHANT_SCENARIOS,
    MARKET,
    SOLVER,
    TIMESTEP_H,
    TIMESTEP_MIN,
    TIMESTEPS_PER_DAY,
    TIMESTEPS_PER_YEAR,
)

logger = logging.getLogger(__name__)


@dataclass
class DispatchInputs:
    da_prices: np.ndarray
    idc_prices: np.ndarray
    afrr_up_capacity_prices: np.ndarray
    afrr_up_activation_prices: np.ndarray
    afrr_down_capacity_prices: np.ndarray
    afrr_down_activation_prices: np.ndarray
    hydro_mw: np.ndarray
    pv_mw: np.ndarray
    pv_installed_mw: float
    grid_limit_mw: float
    contract_share: float
    bess_model: BESSModel
    charging_scenario: str
    afrr_acceptance_factor: np.ndarray
    afrr_up_factor:         np.ndarray
    afrr_down_factor:       np.ndarray


@dataclass
class DispatchResults:
    hydro_to_grid_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hydro_to_bess_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hydro_FiT_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pv_to_grid_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pv_to_bess_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pv_curtailed_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))

    bess_charge_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bess_discharge_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bess_soc_mwh: np.ndarray = field(default_factory=lambda: np.zeros(0))

    bess_sell_da_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bess_buy_da_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bess_sell_idc_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bess_buy_idc_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bess_to_ppa_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))

    afrr_up_capacity_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    afrr_up_activation_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    afrr_down_capacity_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    afrr_down_activation_mw: np.ndarray = field(default_factory=lambda: np.zeros(0))

    ppa_delivered_mwh: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ppa_shortfall_mwh: np.ndarray = field(default_factory=lambda: np.zeros(0))

    revenue_da: np.ndarray = field(default_factory=lambda: np.zeros(0))
    revenue_idc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    revenue_afrr_up_capacity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    revenue_afrr_up_activation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    revenue_afrr_down_capacity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    revenue_afrr_down_activation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    energy_throughput_mwh: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cumulative_throughput: np.ndarray = field(default_factory=lambda: np.zeros(0))

    bess_soc_final_mwh: float = 0.0
    cumulative_cycles_final: float = 0.0
    solver_status: str = "unknown"
    objective_value: float = 0.0


def _simulate_afrr_factors(
    n_steps: int,
    seed: int = 42,
    acceptance_prob: float = 0.80,
    activation_prob: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates three independent annual arrays for the full year.
    acceptance:   Bernoulli(0.80) — 1 if capacity bid accepted, 0 otherwise
    up_factor:    1.0 where accepted AND up activated, else 0
    down_factor:  1.0 where accepted AND down activated, else 0
    Up and down activations are independent Bernoulli(0.25) draws.
    """
    rng = np.random.default_rng(seed)
    acceptance  = rng.binomial(1, acceptance_prob, size=n_steps).astype(float)
    up_active   = rng.binomial(1, activation_prob, size=n_steps).astype(float)
    down_active = rng.binomial(1, activation_prob, size=n_steps).astype(float)
    up_factor   = acceptance * up_active
    down_factor = acceptance * down_active
    return acceptance, up_factor, down_factor


def _build_and_solve(inp: DispatchInputs, time_limit_s: float,
    mip_gap: float | None = None,) -> DispatchResults:
    T = len(inp.da_prices)
    bm = inp.bess_model
    c = inp.contract_share

    scenario = inp.charging_scenario
    
    hydro_is_merchant = scenario in HYDRO_MERCHANT_SCENARIOS
    grid_charging_ok = scenario in GRID_CHARGING_SCENARIOS

    limits = bm.get_dispatch_limits(timestep_minutes=TIMESTEP_MIN)
    E_bess_deg = limits["available_capacity_mwh"]
    Power_MW = bm.params.power_charge_mw
    soc_min = 0.05* E_bess_deg
    soc_max = 0.95* E_bess_deg
    soc_init = bm.energy_mwh
    eta_ch = bm.charge_efficiency
    eta_dis = bm.discharge_efficiency
    prev_power = limits["previous_power_mw"]
    ramp_limit = bm.get_ramp_limit_mw(timestep_minutes=TIMESTEP_MIN)

    window_days = T / TIMESTEPS_PER_DAY
    max_throughput = 2.0 * E_bess_deg * bm.params.max_cycles_per_day * window_days
    max_throughput_per_day = 2.0 * E_bess_deg * bm.params.max_cycles_per_day
    cum_throughput_init = bm.cumulative_throughput_mwh

    afrr_up_res = MARKET.afrr_up_energy_reserve_mwh_per_mw
    afrr_down_res = MARKET.afrr_down_energy_reserve_mwh_per_mw

     
    ppa_penalty = max(500.0, 3.0 * float(np.nanmax(inp.da_prices)))

    m = ConcreteModel()
    m.T = RangeSet(0, T - 1)

    grid_limit = inp.grid_limit_mw

    m.pv_to_grid = Var(m.T, within=NonNegativeReals, bounds=(0, grid_limit))
    m.pv_to_bess = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.pv_curtail = Var(m.T, within=NonNegativeReals)

    m.hydro_to_grid = Var(m.T, within=NonNegativeReals, bounds=(0, grid_limit))
    m.hydro_to_bess = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.hydro_FiT = Var(m.T, within=NonNegativeReals)

    m.bess_buy_da = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.bess_sell_da = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))

    m.bess_sell_idc = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.bess_buy_idc = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    
    m.bess_to_ppa = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))

    m.afrr_up_capacity = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.afrr_down_capacity = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.afrr_up_activation = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.afrr_down_activation = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))

    m.bess_charge = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.bess_discharge = Var(m.T, within=NonNegativeReals, bounds=(0, Power_MW))
    m.soc = Var(m.T, bounds=(soc_min, soc_max))

    m.batt_mode = Var(m.T, within=Binary)
    m.da_bid = Var(m.T, within=Binary)
    m.idc_bid = Var(m.T, within=Binary)
    m.afrr_up_bid  = Var(m.T, within=Binary)   # 1 = up capacity bid placed
    m.afrr_down_bid = Var(m.T, within=Binary)   # 1 = down capacity bid placed

   
    m.throughput_step = Var(m.T, within=NonNegativeReals)
    m.cum_throughput = Var(m.T, within=NonNegativeReals)
    m.ppa_shortfall = Var(m.T, within=NonNegativeReals)

    m.rev_da = Var(m.T, within=Reals)
    m.rev_idc = Var(m.T, within=Reals)
    m.rev_afrr_up_cap = Var(m.T, within=NonNegativeReals)
    m.rev_afrr_up_act = Var(m.T, within=Reals)
    m.rev_afrr_dn_cap = Var(m.T, within=NonNegativeReals)
    m.rev_afrr_dn_act = Var(m.T, within=Reals)

    def pv_balance(model, t):
        return model.pv_to_grid[t] + model.pv_to_bess[t] + model.pv_curtail[t] == float(inp.pv_mw[t])
    m.pv_balance = Constraint(m.T, rule=pv_balance)

    if hydro_is_merchant:
        def hydro_FiT_zero(model, t):
            return model.hydro_FiT[t] == 0.0
        m.hydro_FiT_zero = Constraint(m.T, rule=hydro_FiT_zero)

        def hydro_balance(model, t):
            return model.hydro_to_grid[t] + model.hydro_to_bess[t] <= float(inp.hydro_mw[t])
        m.hydro_balance = Constraint(m.T, rule=hydro_balance)

        def ppa_obligation(model, t):
            obligation = c * (float(inp.pv_mw[t]) + float(inp.hydro_mw[t]))
            delivered = c*(model.pv_to_grid[t] + model.hydro_to_grid[t]) + model.bess_to_ppa[t]
            return model.ppa_shortfall[t] >= obligation - delivered
        m.ppa_obligation = Constraint(m.T, rule=ppa_obligation)
    else:
        def hydro_to_grid_zero(model, t):
            return model.hydro_to_grid[t] == 0.0
        m.hydro_to_grid_zero = Constraint(m.T, rule=hydro_to_grid_zero)

        def hydro_to_bess_zero(model, t):
            return model.hydro_to_bess[t] == 0.0
        m.hydro_to_bess_zero = Constraint(m.T, rule=hydro_to_bess_zero)

        def hydro_FiT_fix(model, t):
            return model.hydro_FiT[t] == min(float(inp.hydro_mw[t]), inp.grid_limit_mw)
        m.hydro_FiT_fix = Constraint(m.T, rule=hydro_FiT_fix)

        def ppa_obligation(model, t):
            obligation = c * float(inp.pv_mw[t])
            delivered = c*model.pv_to_grid[t] + model.bess_to_ppa[t]
            return model.ppa_shortfall[t] >= obligation - delivered
        m.ppa_obligation = Constraint(m.T, rule=ppa_obligation)

    if not grid_charging_ok:
        def no_grid_da(model, t):
            return model.bess_buy_da[t] == 0.0
        m.no_grid_da = Constraint(m.T, rule=no_grid_da)

        def no_grid_idc(model, t):
            return model.bess_buy_idc[t] == 0.0
        m.no_grid_idc = Constraint(m.T, rule=no_grid_idc)

        def no_afrr_down_cap(model, t):
            return model.afrr_down_capacity[t] == 0.0
        m.no_afrr_down_cap = Constraint(m.T, rule=no_afrr_down_cap)
    
    

    #aFRR capacity market constraints:
    # Reserve constraints 
    def afrr_up_reserve(model, t):
        soc_entering = soc_init if t == 0 else model.soc[t - 1]
        return soc_entering >= soc_min + afrr_up_res * model.afrr_up_capacity[t] * float(inp.afrr_acceptance_factor[t])
    m.afrr_up_reserve = Constraint(m.T, rule=afrr_up_reserve)

    def afrr_down_reserve(model, t):
        soc_entering = soc_init if t == 0 else model.soc[t - 1]
        return soc_entering <= soc_max - afrr_down_res * model.afrr_down_capacity[t] * float(inp.afrr_acceptance_factor[t])
    m.afrr_down_reserve = Constraint(m.T, rule=afrr_down_reserve)

    #Capacity bid constraints
    def afrr_up_min(model, t):
        return model.afrr_up_capacity[t] >= 1.0 * model.afrr_up_bid[t]
    m.afrr_up_min = Constraint(m.T, rule=afrr_up_min)

    def afrr_up_max(model, t):
        return model.afrr_up_capacity[t] <= Power_MW * model.afrr_up_bid[t]
    m.afrr_up_max = Constraint(m.T, rule=afrr_up_max)

    def afrr_down_min(model, t):
        return model.afrr_down_capacity[t] >= 1.0 * model.afrr_down_bid[t]
    m.afrr_down_min = Constraint(m.T, rule=afrr_down_min)

    def afrr_down_max(model, t):
        return model.afrr_down_capacity[t] <= Power_MW * model.afrr_down_bid[t]
    m.afrr_down_max = Constraint(m.T, rule=afrr_down_max)


    #afrr activation constraints
    def afrr_up_activation_limit(model, t):
        return model.afrr_up_activation[t] == (model.afrr_up_capacity[t] * float(inp.afrr_up_factor[t]))
    m.afrr_activation_up_limit = Constraint(m.T, rule=afrr_up_activation_limit)

    def afrr_down_activation_limit(model, t):
        return model.afrr_down_activation[t] == (model.afrr_down_capacity[t] * float(inp.afrr_down_factor[t]))
    m.afrr_activation_down_limit = Constraint(m.T, rule=afrr_down_activation_limit)


    #Battery constraints 
    def soc_dyn(model, t):
        if t == 0:
            return model.soc[t] == soc_init + (eta_ch * model.bess_charge[t] - model.bess_discharge[t] / eta_dis) * TIMESTEP_H
        else:
           return model.soc[t] == model.soc[t - 1] + (eta_ch * model.bess_charge[t] - model.bess_discharge[t] / eta_dis) * TIMESTEP_H
    m.soc_dyn = Constraint(m.T, rule=soc_dyn)

    def charge_rule(model, t):
        return model.pv_to_bess[t] + model.hydro_to_bess[t] + model.bess_buy_da[t] + model.bess_buy_idc[t] + model.afrr_down_capacity[t]*float(inp.afrr_acceptance_factor[t]) <= Power_MW
    m.charge_rule = Constraint(m.T, rule=charge_rule)

    def discharge_rule(model, t):
        return model.bess_sell_da[t] + (model.bess_sell_idc[t]) + model.bess_to_ppa[t] + model.afrr_up_capacity[t]*float(inp.afrr_acceptance_factor[t]) <= Power_MW
    m.discharge_rule = Constraint(m.T, rule=discharge_rule)

    def charge_balance(model, t):
         return model.bess_charge[t] == ( 
            model.pv_to_bess[t] + model.hydro_to_bess[t] + model.bess_buy_da[t] + model.bess_buy_idc[t] + model.afrr_down_activation[t])
    m.charge_balance = Constraint(m.T, rule=charge_balance)

    def discharge_balance(model, t):
            return model.bess_discharge[t] == (
        model.bess_sell_da[t] + model.bess_sell_idc[t] + model.bess_to_ppa[t] + model.afrr_up_activation[t])
    m.discharge_balance = Constraint(m.T, rule=discharge_balance)

  
    def charge_mode(model, t):
        return model.bess_charge[t] <= Power_MW * model.batt_mode[t]
    m.charge_mode = Constraint(m.T, rule=charge_mode)

    def discharge_mode(model, t):
        return model.bess_discharge[t] <= Power_MW * (1 - model.batt_mode[t])
    m.discharge_mode = Constraint(m.T, rule=discharge_mode)

    
        
    #Grid limits 
    def export_limit(model, t):
        return (
            model.hydro_FiT[t]
            + model.hydro_to_grid[t]
            + model.pv_to_grid[t]
            + model.bess_sell_da[t]
            + model.bess_sell_idc[t]
            + model.bess_to_ppa[t]
            + model.afrr_up_activation[t]
        ) <= grid_limit
    m.export_limit = Constraint(m.T, rule=export_limit)

    def import_limit(model, t):
        return ( 
            model.bess_buy_da[t]
            + model.bess_buy_idc[t]
            + model.afrr_down_activation[t]
        ) <= grid_limit
    m.import_limit = Constraint(m.T, rule=import_limit)

    
    
    # Binary constraints 
    def da_buy_flag(model, t):
        return model.bess_buy_da[t] <= Power_MW * model.da_bid[t]
    m.da_buy_flag = Constraint(m.T, rule=da_buy_flag)

    def da_sell_flag(model, t):
        return model.bess_sell_da[t] <= Power_MW * (1 - model.da_bid[t])
    m.da_sell_flag = Constraint(m.T, rule=da_sell_flag)

    def idc_buy_flag(model, t):
        return model.bess_buy_idc[t] <= Power_MW * model.idc_bid[t]
    m.idc_buy_flag = Constraint(m.T, rule=idc_buy_flag)

    def idc_sell_flag(model, t):
        return model.bess_sell_idc[t] <= Power_MW * (1 - model.idc_bid[t])
    m.idc_sell_flag = Constraint(m.T, rule=idc_sell_flag)

 
    def afrr_up_needs_discharge_mode(model, t):
        return model.afrr_up_bid[t] <= 1 - model.batt_mode[t]
    m.afrr_up_needs_discharge_mode = Constraint(m.T, rule=afrr_up_needs_discharge_mode)

    def afrr_down_needs_charge_mode(model, t):
        return model.afrr_down_bid[t] <= model.batt_mode[t]
    m.afrr_down_needs_charge_mode = Constraint(m.T, rule=afrr_down_needs_charge_mode)


    #Battery specific constraints
    def ramp_up(model, t):
        net_t = model.bess_charge[t] - model.bess_discharge[t]
        net_prev = prev_power if t == 0 else model.bess_charge[t - 1] - model.bess_discharge[t - 1]
        return net_t - net_prev <= ramp_limit
    m.ramp_up = Constraint(m.T, rule=ramp_up)

    def ramp_down(model, t):
        net_t = model.bess_charge[t] - model.bess_discharge[t]
        net_prev = prev_power if t == 0 else model.bess_charge[t - 1] - model.bess_discharge[t - 1]
        return net_prev - net_t <= ramp_limit
    m.ramp_down = Constraint(m.T, rule=ramp_down)

    def throughput_step_rule(model, t):
        return model.throughput_step[t] == (model.bess_charge[t] + model.bess_discharge[t]) * TIMESTEP_H
    m.throughput_step_def = Constraint(m.T, rule=throughput_step_rule)

    def cum_throughput_rule(model, t):
        prev = cum_throughput_init if t == 0 else model.cum_throughput[t - 1]
        return model.cum_throughput[t] == prev + model.throughput_step[t]
    m.cum_throughput_def = Constraint(m.T, rule=cum_throughput_rule)

    def cycle_budget_rule(model):
        return model.cum_throughput[T - 1] <= cum_throughput_init + max_throughput
    m.cycle_budget = Constraint(rule=cycle_budget_rule)

    def daily_cycle_budget_rule(model, t):
        day_start = (t // TIMESTEPS_PER_DAY) * TIMESTEPS_PER_DAY
        baseline = cum_throughput_init if day_start == 0 else model.cum_throughput[day_start - 1]
        return model.cum_throughput[t] - baseline <= max_throughput_per_day
    m.daily_cycle_budget = Constraint(m.T, rule=daily_cycle_budget_rule)


    
    # Revenues constraints 
    def rev_da_rule(model, t):
        merchant_gen_da = (model.hydro_to_grid[t] + model.pv_to_grid[t]) * (1 - c)
        battery_da = model.bess_sell_da[t] - model.bess_buy_da[t]
        return model.rev_da[t] == inp.da_prices[t] * (merchant_gen_da + battery_da) * TIMESTEP_H
    m.rev_da_def = Constraint(m.T, rule=rev_da_rule)

    def rev_idc_rule(model, t):
        return model.rev_idc[t] == inp.idc_prices[t] * (model.bess_sell_idc[t] - model.bess_buy_idc[t]) * TIMESTEP_H
    m.rev_idc_def = Constraint(m.T, rule=rev_idc_rule)

    def rev_afrr_up_cap_rule(model, t):
        return model.rev_afrr_up_cap[t] == inp.afrr_up_capacity_prices[t] * model.afrr_up_capacity[t] *float(inp.afrr_acceptance_factor[t]) 
    m.rev_afrr_up_cap_def = Constraint(m.T, rule=rev_afrr_up_cap_rule)

    def rev_afrr_dn_cap_rule(model, t):
        return model.rev_afrr_dn_cap[t] == inp.afrr_down_capacity_prices[t] * model.afrr_down_capacity[t] *float(inp.afrr_acceptance_factor[t]) 
    m.rev_afrr_dn_cap_def = Constraint(m.T, rule=rev_afrr_dn_cap_rule)

    def rev_afrr_up_act_rule(model, t):
         return model.rev_afrr_up_act[t] == inp.afrr_up_activation_prices[t] * model.afrr_up_activation[t] * TIMESTEP_H 
    m.rev_afrr_up_act_def = Constraint(m.T, rule=rev_afrr_up_act_rule)

    def rev_afrr_dn_act_rule(model, t):
        return model.rev_afrr_dn_act[t] == inp.afrr_down_activation_prices[t] * model.afrr_down_activation[t] * TIMESTEP_H 
    m.rev_afrr_dn_act_def = Constraint(m.T, rule=rev_afrr_dn_act_rule)

  
    def total_obj(model):
        merchant_rev = sum(
            model.rev_da[t] + model.rev_idc[t] + model.rev_afrr_up_cap[t] + model.rev_afrr_up_act[t] + model.rev_afrr_dn_cap[t] + model.rev_afrr_dn_act[t] 
            for t in model.T
        )
        shortfall_cost = ppa_penalty * sum(model.ppa_shortfall[t] for t in model.T)* TIMESTEP_H
        return merchant_rev - shortfall_cost
    m.obj = Objective(rule=total_obj, sense=maximize)

    opt = SolverFactory(SOLVER.solver_name)
    if not opt.available():
        raise RuntimeError(f"Solver '{SOLVER.solver_name}' not available.")

    solver_name = SOLVER.solver_name.lower()

    if "gurobi" in solver_name:
        opt.options["TimeLimit"] = float(time_limit_s)
        opt.options["OutputFlag"] = int(SOLVER.output_flag)
        if mip_gap is not None:
            opt.options["MIPGap"] = float(mip_gap)

    elif "cbc" in solver_name:
        opt.options["seconds"] = float(time_limit_s)
        if mip_gap is not None:
            opt.options["ratio"] = float(mip_gap)

    elif "highs" in solver_name:
        opt.options["time_limit"] = float(time_limit_s)
        if mip_gap is not None:
            opt.options["mip_rel_gap"] = float(mip_gap)

    result = opt.solve(m, tee=(SOLVER.output_flag == 1))

    term = result.solver.termination_condition
    status = result.solver.status

    acceptable_terms = {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.maxTimeLimit,
    }

    if term not in acceptable_terms:
        raise RuntimeError(
            f"Solver failed — status: {status} | termination: {term}"
        )

    obj_val = float(value(m.obj))

    ch_arr = np.array([value(m.bess_charge[t]) for t in range(T)])
    dis_arr = np.array([value(m.bess_discharge[t]) for t in range(T)])

    optimizer_final_soc = float(value(m.soc[T - 1]))



    res = DispatchResults(
        hydro_to_grid_mw=np.array([value(m.hydro_to_grid[t]) for t in range(T)]),
        hydro_to_bess_mw=np.array([value(m.hydro_to_bess[t]) for t in range(T)]),
        hydro_FiT_mw=np.array([value(m.hydro_FiT[t]) for t in range(T)]),
        pv_to_grid_mw=np.array([value(m.pv_to_grid[t]) for t in range(T)]),
        pv_to_bess_mw=np.array([value(m.pv_to_bess[t]) for t in range(T)]),
        pv_curtailed_mw=np.array([value(m.pv_curtail[t]) for t in range(T)]),
        bess_charge_mw=ch_arr,
        bess_discharge_mw=dis_arr,
        bess_soc_mwh=np.array([value(m.soc[t]) for t in range(T)]),
        bess_sell_da_mw=np.array([value(m.bess_sell_da[t]) for t in range(T)]),
        bess_buy_da_mw=np.array([value(m.bess_buy_da[t]) for t in range(T)]),
        bess_sell_idc_mw=np.array([value(m.bess_sell_idc[t]) for t in range(T)]),
        bess_buy_idc_mw=np.array([value(m.bess_buy_idc[t]) for t in range(T)]),
        bess_to_ppa_mw=np.array([value(m.bess_to_ppa[t]) for t in range(T)]),
        afrr_up_capacity_mw=np.array([value(m.afrr_up_capacity[t])*float(inp.afrr_acceptance_factor[t])  for t in range(T)]),
        afrr_down_capacity_mw=np.array([value(m.afrr_down_capacity[t])*float(inp.afrr_acceptance_factor[t]) for t in range(T)]),
        afrr_up_activation_mw=np.array([value(m.afrr_up_activation[t]) for t in range(T)]),
        afrr_down_activation_mw=np.array([value(m.afrr_down_activation[t])  for t in range(T)]),
        ppa_shortfall_mwh=np.array([value(m.ppa_shortfall[t]) * TIMESTEP_H for t in range(T)]),
        ppa_delivered_mwh=np.array([
            (c*(value(m.pv_to_grid[t]) + value(m.hydro_to_grid[t])) + value(m.bess_to_ppa[t])) * TIMESTEP_H for t in range(T)
        ]),
        revenue_da=np.array([value(m.rev_da[t]) for t in range(T)]),
        revenue_idc=np.array([value(m.rev_idc[t]) for t in range(T)]),
        revenue_afrr_up_capacity=np.array([value(m.rev_afrr_up_cap[t]) for t in range(T)]),
        revenue_afrr_up_activation=np.array([value(m.rev_afrr_up_act[t]) for t in range(T)]),
        revenue_afrr_down_capacity=np.array([value(m.rev_afrr_dn_cap[t]) for t in range(T)]),
        revenue_afrr_down_activation=np.array([value(m.rev_afrr_dn_act[t]) for t in range(T)]),
        energy_throughput_mwh=np.array([value(m.throughput_step[t]) for t in range(T)]),
        cumulative_throughput=np.array([value(m.cum_throughput[t]) for t in range(T)]),
        solver_status=str(result.solver.termination_condition),
        objective_value=obj_val,
    )

    bm.simulate_from_dispatch(ch_arr, dis_arr, timestep_minutes=TIMESTEP_MIN)
    res.bess_soc_final_mwh = optimizer_final_soc
    res.cumulative_cycles_final = bm.cumulative_cycles
    return res


def run_annual_dispatch(
    prices: Dict[str, np.ndarray],
    pv_profile: np.ndarray,
    hydro_profile: np.ndarray,
    bess_model: BESSModel,
    pv_installed_mw: float,
    grid_limit_mw: float,
    contract_share: float,
    charging_scenario: str,
    window_steps: int = 7 * TIMESTEPS_PER_DAY,
    max_total_runtime_s: float = 180.0,
    target_mip_gap: float = 0.01,
) -> Dict:
    if grid_limit_mw is None:
        grid_limit_mw = HYDRO.grid_injection_capacity_mw

    n_steps = TIMESTEPS_PER_YEAR
    n_windows = int(np.ceil(n_steps / window_steps))
    window_results: list[DispatchResults] = []

    annual_acceptance, annual_up_factor, annual_down_factor = _simulate_afrr_factors(TIMESTEPS_PER_YEAR, seed=42)

    for w in range(n_windows):
        t0 = w * window_steps
        t1 = min(t0 + window_steps, n_steps)
        idx = slice(t0, t1)

        window_time_limit_s = max_total_runtime_s / n_windows

        win_inp = DispatchInputs(
            da_prices=prices["da"][idx],
            idc_prices=prices["idc"][idx],
            afrr_up_capacity_prices=prices["afrr_up_capacity"][idx],
            afrr_up_activation_prices=prices["afrr_up_activation"][idx],
            afrr_down_capacity_prices=prices["afrr_down_capacity"][idx],
            afrr_down_activation_prices=prices["afrr_down_activation"][idx],
            hydro_mw=hydro_profile[idx],
            pv_mw=pv_profile[idx],
            pv_installed_mw=pv_installed_mw,
            grid_limit_mw=grid_limit_mw,
            contract_share=contract_share,
            bess_model=bess_model,
            charging_scenario=charging_scenario,
            afrr_acceptance_factor = annual_acceptance[idx],
            afrr_up_factor = annual_up_factor[idx],
            afrr_down_factor = annual_down_factor[idx],
        )
       # window_results.append(_build_and_solve(win_inp))
        window_results.append( _build_and_solve( win_inp, time_limit_s=window_time_limit_s, mip_gap=target_mip_gap, )
)


    def _cat(attr: str) -> np.ndarray:
        return np.concatenate([getattr(r, attr) for r in window_results])

    hydro_grid = _cat("hydro_to_grid_mw")
    hydro_FiT = _cat("hydro_FiT_mw")
    hydro_bess = _cat("hydro_to_bess_mw")
    pv_grid = _cat("pv_to_grid_mw")
    bess_sell_da = _cat("bess_sell_da_mw")
    bess_sell_idc = _cat("bess_sell_idc_mw")
    bess_buy_idc = _cat("bess_buy_idc_mw")
    bess_buy_da = _cat("bess_buy_da_mw")
    bess_to_ppa = _cat("bess_to_ppa_mw")
    bess_dis = _cat("bess_discharge_mw")
    bess_ch = _cat("bess_charge_mw")
    total_export_mw = hydro_FiT + hydro_grid + pv_grid + bess_sell_da + bess_to_ppa + bess_sell_idc

    annual_revenues = {
        "da": float(_cat("revenue_da").sum()),
        "idc": float(_cat("revenue_idc").sum()),
        "afrr_up_capacity": float(_cat("revenue_afrr_up_capacity").sum()),
        "afrr_up_activation": float(_cat("revenue_afrr_up_activation").sum()),
        "afrr_down_capacity": float(_cat("revenue_afrr_down_capacity").sum()),
        "afrr_down_activation": float(_cat("revenue_afrr_down_activation").sum()),
    }

    return {
        "windows": window_results,
        "annual_revenues": annual_revenues,
        "total_output_mwh": float(total_export_mw.sum() * TIMESTEP_H),
        "hydro_annual_mwh": float((hydro_FiT + hydro_grid).sum() * TIMESTEP_H),
        "pv_annual_mwh": float(pv_grid.sum() * TIMESTEP_H),
        "bess_discharge_mwh": float(bess_dis.sum() * TIMESTEP_H),
        "bess_charge_mwh": float(bess_ch.sum() * TIMESTEP_H),
        "hydro_to_bess_mwh": float(hydro_bess.sum() * TIMESTEP_H),
        "pv_curtailed_mwh": float(_cat("pv_curtailed_mw").sum() * TIMESTEP_H),
        "ppa_delivered_mwh": float(_cat("ppa_delivered_mwh").sum()),
        "ppa_shortfall_mwh": float(_cat("ppa_shortfall_mwh").sum()),
        "final_soc_mwh": bess_model.energy_mwh,
        "total_cycles": bess_model.cumulative_cycles,
        "final_degradation": bess_model.degradation_factor,
        "total_throughput_mwh": bess_model.cumulative_throughput_mwh,
        "charging_scenario": charging_scenario,
        "total_output_mw": total_export_mw,
        "hydro_mw": hydro_FiT + hydro_grid,
        "hydro_to_bess_mw": hydro_bess,
        "pv_to_grid_mw": pv_grid,
        "bess_discharge_mw": bess_dis,
        "bess_charge_mw": bess_ch,
        "bess_soc_mwh": _cat("bess_soc_mwh"),
        "bess_sell_da_mw": bess_sell_da,
        "bess_sell_idc_mw": bess_sell_idc,
        "bess_buy_da_mw": bess_buy_da,
        "bess_buy_idc_mw": bess_buy_idc,
        "bess_to_ppa_mw": bess_to_ppa,
    }