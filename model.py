"""
Created on Tue Jan  7 21:10:18 2025

@author: ckadelka

Drug-Induced Plasticity (DIP) Extension
========================================
The s→r phenotypic transition rate is now therapy-state-dependent:

    t_{s,eff}(t) = t_s + δ·θ(t)

where:
    t_s  : basal (off-treatment) s→r phenotypic transition rate
    δ    : DIP parameter — drug-induced additional s→r rate (δ ≥ 0)
    θ(t) : drug indicator  {1 if treatment on,  0 if treatment off}

The r→s rate t_r is UNCHANGED by treatment.

Treatment hysteresis:
    θ: 0 → 1  when  S + R > threshold_treatment_on  · K
    θ: 1 → 0  when  S + R < threshold_treatment_off · K
    Between thresholds θ maintains its previous state.

Two drug action modes (controlled via r_s_treatment and d_s_treatment):
    Cytotoxic   (ε=1,    d_th>0):  d_s_treatment = d_th,  r_s_treatment = r_s
    Cytostatic  (ε∈(0,1), d_th=0): d_s_treatment = 0,     r_s_treatment = ε·r_s

Both modes can carry DIP (delta > 0). Set delta = 0 to recover the original model.

Constraint: t_s + δ ≤ r_s  (net DIP rate cannot exceed basal growth rate)
Parameter ranges (Gevertz data):
    μ_s (r_s)  : [0.00247, 0.05]
    μ_r (r_r)  : [0.00297, 0.05]
    t_s, t_r   : [0.000125, 0.0015]
    α_rs (c)   : [0.1, 7.4]
    α_sr (d)   : [0.1, 7.4]
    δ (delta)  : [0.000125, 0.25]
    ε          : 0.33 or 0.1
    d_th       : 0.01 or 0.05
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def competition_transition_model(X, t, r_s, r_r, c, d, K, t_s, t_r):
    """
    Baseline 2-species logistic growth model with asymmetric competition
    and phenotypic transition. No treatment.

    Parameters
    ----------
    X   : array-like [s, r]
          s : sensitive cell population
          r : resistant cell population
    t   : float — current time
    r_s : float — growth rate of sensitive cells (μ_s)
    r_r : float — growth rate of resistant cells (μ_r)
    c   : float — competitive effect of resistant on sensitive (α_rs)
    d   : float — competitive effect of sensitive on resistant (α_sr)
    K   : float — carrying capacity
    t_s : float — basal s→r phenotypic transition rate
    t_r : float — r→s phenotypic transition rate

    Returns
    -------
    dX : array-like [ds/dt, dr/dt]
    """
    s, r = X
    dX = np.zeros(2, dtype=np.float64)
    dX[0] = r_s * s * (1 - (s + c * r) / K) - t_s * s + t_r * r
    dX[1] = r_r * r * (1 - (r + d * s) / K) + t_s * s - t_r * r
    return dX


@jit(nopython=True)
def competition_transition_model_adaptive_treatment(X, t, r_s, r_r, c, d, K, t_s, t_r, d_s, d_r):
    """
    ODE for the DIP-extended model. Parameters passed in are already the
    effective (θ-modulated) values; θ-switching logic lives in RK4_adaptive_therapy.

    When treatment is off (θ=0): r_s, t_s, d_s are basal values.
    When treatment is on  (θ=1): r_s → r_s_treatment,
                                  t_s → t_s + delta  (DIP),
                                  d_s → d_s_treatment.

    Parameters
    ----------
    X   : array-like [s, r]
    t   : float — current time
    r_s : float — effective growth rate of sensitive cells
    r_r : float — growth rate of resistant cells (invariant)
    c   : float — competitive effect of resistant on sensitive (α_rs)
    d   : float — competitive effect of sensitive on resistant (α_sr)
    K   : float — carrying capacity
    t_s : float — effective s→r transition rate (= t_s or t_s + delta)
    t_r : float — r→s transition rate (invariant)
    d_s : float — effective death rate of sensitive cells
    d_r : float — death rate of resistant cells

    Returns
    -------
    dX : array-like [ds/dt, dr/dt]
    """
    s, r = X
    dX = np.zeros(2, dtype=np.float64)
    dX[0] = r_s * s * (1 - (s + c * r) / K) - t_s * s + t_r * r - d_s * s
    dX[1] = r_r * r * (1 - (r + d * s) / K) + t_s * s - t_r * r - d_r * r
    return dX


@jit(nopython=True)
def RK4(func, X0, ts, r_s, r_r, c, d, K, t_s, t_r, dt):
    """
    Fourth-order Runge-Kutta solver for the no-treatment model.

    Parameters
    ----------
    func : callable — ODE function (competition_transition_model)
    X0   : array-like [s0, r0] — initial state
    ts   : array-like — time points
    r_s, r_r, c, d, K, t_s, t_r : model parameters (see competition_transition_model)
    dt   : float — time step size

    Returns
    -------
    X : ndarray (nt, 2) — trajectories [s(t), r(t)]
    """
    nt = len(ts)
    X = np.zeros((nt, 2), dtype=np.float64)
    X[0, :] = X0

    for i in range(nt - 1):
        k1 = func(X[i], ts[i], r_s, r_r, c, d, K, t_s, t_r)
        k2 = func(X[i] + dt*k1/2., ts[i] + dt/2., r_s, r_r, c, d, K, t_s, t_r)
        k3 = func(X[i] + dt*k2/2., ts[i] + dt/2., r_s, r_r, c, d, K, t_s, t_r)
        k4 = func(X[i] + dt*k3, ts[i] + dt, r_s, r_r, c, d, K, t_s, t_r)
        X[i+1] = X[i] + dt / 6. * (k1 + 2.*k2 + 2.*k3 + k4)
    return X


@jit(nopython=True)
def RK4_adaptive_therapy(func, X0, ts,
                          r_s, r_s_treatment, r_r,
                          c, d, K,
                          t_s, delta, t_r,
                          d_s, d_s_treatment, d_r,
                          threshold_treatment_on, threshold_treatment_off,
                          dt):
    """
    RK4 solver for the DIP-extended adaptive (or constant-dose) therapy model.

    Core DIP logic
    --------------
    When θ(t) = 0 (off-treatment):  t_{s,eff} = t_s
    When θ(t) = 1 (on-treatment):   t_{s,eff} = t_s + delta

    Treatment switching (hysteresis)
    --------------------------------
    θ: 0 → 1  if  S + R > threshold_treatment_on  · K
    θ: 1 → 0  if  S + R < threshold_treatment_off · K
    Otherwise: θ unchanged (hysteresis zone).

    For constant-dose therapy (CDT): set threshold_treatment_on = 0 and
    threshold_treatment_off = np.nan — treatment turns on immediately and
    never turns off (np.nan comparisons always return False).

    Parameters
    ----------
    func            : callable — ODE function (competition_transition_model_adaptive_treatment)
    X0              : array-like [s0, r0] — initial state
    ts              : array-like — time points
    r_s             : float — basal growth rate of sensitive cells (off-treatment, μ_s)
    r_s_treatment   : float — effective growth rate of sensitive cells under treatment
                      Cytotoxic:  r_s_treatment = r_s        (ε = 1)
                      Cytostatic: r_s_treatment = ε · r_s    (ε ∈ (0, 1))
    r_r             : float — growth rate of resistant cells (invariant, μ_r)
    c               : float — competitive effect of resistant on sensitive (α_rs)
    d               : float — competitive effect of sensitive on resistant (α_sr)
    K               : float — carrying capacity
    t_s             : float — basal s→r phenotypic transition rate (off-treatment)
    delta           : float — DIP parameter. Drug-induced additional s→r transition rate.
                      t_{s,eff} = t_s + delta when θ = 1; t_s when θ = 0.
                      Constraint: t_s + delta ≤ r_s.
                      Set delta = 0 to disable DIP (recovers original model).
    t_r             : float — r→s phenotypic transition rate (invariant)
    d_s             : float — basal death rate of sensitive cells (typically 0 off-treatment)
    d_s_treatment   : float — death rate of sensitive cells under treatment (d_th)
                      Cytotoxic:  d_s_treatment = d_th > 0  (e.g., 0.01 or 0.05)
                      Cytostatic: d_s_treatment = 0
    d_r             : float — death rate of resistant cells
    threshold_treatment_on  : float — fraction of K at which treatment turns on
    threshold_treatment_off : float — fraction of K at which treatment turns off
    dt              : float — time step size

    Returns
    -------
    X         : ndarray (nt, 2) — trajectories [s(t), r(t)]
    treatment : ndarray (nt,) int16 — θ(t): 1 when treatment on, 0 when off
    """
    nt = len(ts)
    X = np.zeros((nt, 2), dtype=np.float64)
    X[0, :] = X0

    # Initialise with off-treatment (θ = 0) effective parameters
    r_s_current = r_s
    t_s_current = t_s       # basal s→r rate; DIP will add delta when θ = 1
    d_s_current = d_s
    treatment = np.zeros(nt, dtype=np.int16)

    for i in range(nt - 1):
        k1 = func(X[i], ts[i],
                  r_s_current, r_r, c, d, K, t_s_current, t_r, d_s_current, d_r)
        k2 = func(X[i] + dt*k1/2., ts[i] + dt/2.,
                  r_s_current, r_r, c, d, K, t_s_current, t_r, d_s_current, d_r)
        k3 = func(X[i] + dt*k2/2., ts[i] + dt/2.,
                  r_s_current, r_r, c, d, K, t_s_current, t_r, d_s_current, d_r)
        k4 = func(X[i] + dt*k3, ts[i] + dt,
                  r_s_current, r_r, c, d, K, t_s_current, t_r, d_s_current, d_r)
        X[i+1] = X[i] + dt / 6. * (k1 + 2.*k2 + 2.*k3 + k4)

        total = X[i+1, 0] + X[i+1, 1]

        if total > threshold_treatment_on * K and treatment[i] == 0:
            # θ: 0 → 1  (turn treatment on)
            r_s_current = r_s_treatment
            t_s_current = t_s + delta       # DIP: drug-induced uplift to s→r rate
            d_s_current = d_s_treatment
            treatment[i+1] = 1

        elif total < threshold_treatment_off * K and treatment[i] == 1:
            # θ: 1 → 0  (turn treatment off)
            r_s_current = r_s
            t_s_current = t_s               # revert to basal s→r rate
            d_s_current = d_s
            treatment[i+1] = 0

        else:
            # Hysteresis: θ unchanged
            treatment[i+1] = treatment[i]

    return X, treatment


def simulate(r_s=1, r_r=1, c=0.2, d=0.2, K=10000,
             t_s=0.02, t_r=0.02,
             s0=100, r0=100, dt=0.1, t_end=500):
    """
    Simulate the baseline (no-treatment) competition-transition model.

    Parameters
    ----------
    r_s, r_r : float — growth rates of sensitive/resistant cells
    c        : float — competitive effect of resistant on sensitive (α_rs)
    d        : float — competitive effect of sensitive on resistant (α_sr)
    K        : float — carrying capacity
    t_s      : float — s→r phenotypic transition rate
    t_r      : float — r→s phenotypic transition rate
    s0, r0   : float — initial cell counts
    dt       : float — time step size
    t_end    : float — simulation end time

    Returns
    -------
    ts       : ndarray — time points
    solution : ndarray (nt, 2) — trajectories [s(t), r(t)]
    params   : dict — parameters used
    """
    ts = np.linspace(0, t_end, round(t_end / dt) + 1)
    x0 = np.array([s0, r0], dtype=np.float64)
    params = {'r_s': r_s, 'r_r': r_r, 'c': c, 'd': d, 'K': K,
              't_s': t_s, 't_r': t_r, 'dt': dt}

    # Resolve any string-linked parameters
    n_reduced = 1
    while n_reduced > 0:
        n_reduced = 0
        for key in params:
            if type(params[key]) == str and type(params[params[key]]) in [float, int, bool, np.float64]:
                params[key] = params[params[key]]
                n_reduced += 1

    r_s = params['r_s']
    r_r = params['r_r']
    c   = params['c']
    d   = params['d']
    K   = params['K']
    t_s = params['t_s']
    t_r = params['t_r']
    dt  = params['dt']

    solution = RK4(competition_transition_model, x0, ts,
                   r_s, r_r, c, d, K, t_s, t_r, dt)
    return ts, solution, params


def simulate_adaptive_therapy(r_s=0.05, r_s_treatment=0.01, r_r=0.05,
                              c=0.2, d=0.2, K=10000,
                              t_s=0.005, delta=0., t_r=0.005,
                              d_s=0, d_s_treatment=0.01, d_r=0,
                              threshold_treatment_on=0.5, threshold_treatment_off=0.1,
                              s0=100, r0=100,
                              dt=0.1, t_end=2500):
    """
    Simulate the DIP-extended adaptive (or constant-dose) therapy model.

    Drug-Induced Plasticity: under treatment, the s→r phenotypic transition
    rate becomes t_s + delta instead of t_s. Set delta = 0 to recover the
    original model exactly.

    Parameters
    ----------
    r_s             : float — basal growth rate of sensitive cells (μ_s)
    r_s_treatment   : float — effective growth rate of sensitive cells under treatment
                      Cytotoxic:  r_s_treatment = r_s        (ε = 1)
                      Cytostatic: r_s_treatment = ε · r_s    (ε ∈ (0, 1), e.g. 0.33 or 0.1)
    r_r             : float — growth rate of resistant cells (μ_r)
    c               : float — competitive effect of resistant on sensitive (α_rs)
    d               : float — competitive effect of sensitive on resistant (α_sr)
    K               : float — carrying capacity (default 10,000)
    t_s             : float — basal s→r phenotypic transition rate (off-treatment)
    delta           : float — DIP parameter. Drug-induced additional s→r transition rate.
                      t_{s,eff} = t_s + delta when θ = 1; t_s when θ = 0.
                      Constraint: t_s + delta ≤ r_s.
                      Suggested range: [0.000125, 0.25] (Gevertz data).
                      Set delta = 0 to disable DIP.
    t_r             : float — r→s phenotypic transition rate (invariant)
    d_s             : float — basal death rate of sensitive cells (typically 0)
    d_s_treatment   : float — death rate of sensitive cells under treatment (d_th)
                      Cytotoxic:  d_s_treatment > 0  (e.g. 0.01 or 0.05)
                      Cytostatic: d_s_treatment = 0
    d_r             : float — death rate of resistant cells
    threshold_treatment_on  : float — fraction of K above which treatment starts
    threshold_treatment_off : float — fraction of K below which treatment stops
    s0, r0          : float — initial cell counts
    dt              : float — time step size
    t_end           : float — simulation end time

    Returns
    -------
    ts        : ndarray — time points
    solution  : ndarray (nt, 2) — trajectories [s(t), r(t)]
    treatment : ndarray (nt,) — binary treatment schedule θ(t)
    params    : dict — parameters used
    """
    ts = np.linspace(0, t_end, round(t_end / dt) + 1)
    x0 = np.array([s0, r0], dtype=np.float64)
    params = {
        'r_s': r_s, 'r_s_treatment': r_s_treatment, 'r_r': r_r,
        'c': c, 'd': d, 'K': K,
        't_s': t_s, 'delta': delta, 't_r': t_r,
        'd_s': d_s, 'd_s_treatment': d_s_treatment, 'd_r': d_r,
        'threshold_treatment_on': threshold_treatment_on,
        'threshold_treatment_off': threshold_treatment_off,
        'dt': dt
    }

    # Resolve any string-linked parameters
    n_reduced = 1
    while n_reduced > 0:
        n_reduced = 0
        for key in params:
            if type(params[key]) == str and type(params[params[key]]) in [float, int, bool, np.float64]:
                params[key] = params[params[key]]
                n_reduced += 1

    r_s                    = params['r_s']
    r_s_treatment          = params['r_s_treatment']
    r_r                    = params['r_r']
    c                      = params['c']
    d                      = params['d']
    K                      = params['K']
    t_s                    = params['t_s']
    delta                  = params['delta']
    t_r                    = params['t_r']
    d_s                    = params['d_s']
    d_s_treatment          = params['d_s_treatment']
    d_r                    = params['d_r']
    threshold_treatment_on  = params['threshold_treatment_on']
    threshold_treatment_off = params['threshold_treatment_off']
    dt                     = params['dt']

    solution, treatment = RK4_adaptive_therapy(
        competition_transition_model_adaptive_treatment,
        x0, ts,
        r_s, r_s_treatment, r_r,
        c, d, K,
        t_s, delta, t_r,
        d_s, d_s_treatment, d_r,
        threshold_treatment_on, threshold_treatment_off,
        dt
    )
    return ts, solution, treatment, params


def plot_ttp_vs_delta(delta_values, progression_fraction=0.8, t_end=3000, folder_name='figs', **base_params):
    """
    Plots Time to Progression (TTP) for Constant Dose Therapy (CDT)
    and Adaptive Therapy (AT) as a function of the DIP parameter delta.
    
    Progression is defined as the total cell population (s + r) 
    exceeding a specified fraction of the carrying capacity (K).
    """
    import matplotlib.pyplot as plt
    import os
    
    if not os.path.exists(folder_name) and len(folder_name) > 0:
        os.makedirs(folder_name)
        
    ttp_cdt = []
    ttp_at = []
    
    # Define the absolute population threshold for progression
    K = base_params.get('K', 10000)
    progression_threshold = progression_fraction * K

    for delta in delta_values:
        # 1. Simulate Constant Dose Therapy (CDT)
        # We merge base_params with the specific CDT overrides
        cdt_params = {
            **base_params,
            'delta': delta,
            'threshold_treatment_on': 0.0,
            'threshold_treatment_off': np.nan,
            't_end': t_end
        }
        
        ts_cdt, res_cdt, _, _ = simulate_adaptive_therapy(**cdt_params)
        total_cells_cdt = res_cdt[:, 0] + res_cdt[:, 1]
        
        # Find the first time point where total cells exceed the threshold
        idx_cdt = np.where(total_cells_cdt > progression_threshold)[0]
        if len(idx_cdt) > 0:
            ttp_cdt.append(ts_cdt[idx_cdt[0]])
        else:
            ttp_cdt.append(np.nan) # Tumor did not progress within t_end

        # 2. Simulate Adaptive Therapy (AT)
        # We merge base_params (which already has the AT thresholds) with delta and t_end
        at_params = {
            **base_params,
            'delta': delta,
            't_end': t_end
        }
        
        ts_at, res_at, _, _ = simulate_adaptive_therapy(**at_params)
        total_cells_at = res_at[:, 0] + res_at[:, 1]
        
        idx_at = np.where(total_cells_at > progression_threshold)[0]
        if len(idx_at) > 0:
            ttp_at.append(ts_at[idx_at[0]])
        else:
            ttp_at.append(np.nan)

    # Generate the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Use masks to avoid plotting np.nan lines where progression wasn't reached
    mask_cdt = np.isfinite(ttp_cdt)
    mask_at = np.isfinite(ttp_at)
    
    ax.plot(delta_values[mask_cdt], np.array(ttp_cdt)[mask_cdt], 
            marker='o', linestyle='-', color='blue', label='Constant Dose Therapy')
    ax.plot(delta_values[mask_at], np.array(ttp_at)[mask_at], 
            marker='s', linestyle='-', color='red', label='Adaptive Therapy')

    ax.set_xlabel(r'Drug-Induced Plasticity rate ($\delta$)')
    ax.set_ylabel(f'Time to Progression ($S+R > {progression_fraction}K$)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'TTP_vs_Delta.pdf'), bbox_inches='tight')
    plt.show()


def plot_time_series(r_s=1, r_s_treatment=0.5, r_r=1,
                     c=0.2, d=0.2, K=10000,
                     t_s=0.02, delta=0., t_r=0.02,
                     d_s=0, d_s_treatment=0.01, d_r=0,
                     threshold_treatment_on=0.5, threshold_treatment_off=0.1,
                     s0=1000, r0=1000,
                     dt=1, t_end=500, folder_name='figs'):
    import matplotlib.pyplot as plt
    from utils import params_to_text, get_suffix
    import os

    if not os.path.exists(folder_name) and len(folder_name) > 0:
        os.makedirs(folder_name)

    ts, results, treatment, params = simulate_adaptive_therapy(
        r_s=r_s, r_s_treatment=r_s_treatment, r_r=r_r,
        c=c, d=d, K=K,
        t_s=t_s, delta=delta, t_r=t_r,
        d_s=d_s, d_s_treatment=d_s_treatment, d_r=d_r,
        threshold_treatment_on=threshold_treatment_on,
        threshold_treatment_off=threshold_treatment_off,
        s0=s0, r0=r0, dt=dt, t_end=t_end
    )
    s = results[:, 0]
    r = results[:, 1]
    proportion_sensitive_cells = s / (s + r)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(ts, r, linestyle='-', label='resistant')
    ax.plot(ts, s, linestyle='-', label='sensitive')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax2 = ax.twinx()
    ax.spines[['top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)
    ax2.plot(ts, proportion_sensitive_cells, color='k', linestyle='--')
    dummy = np.where(treatment == 1)[0]
    [y1, y2] = ax2.get_ylim()
    ax2.plot(ts[dummy], [y1 + 0.01 * (y2 - y1)] * len(dummy), 'ko')
    ax.set_xlabel('time')
    ax.set_ylabel('number of cells')
    ax2.set_ylabel('proportion sensitive cells')
    text_left, text_right = params_to_text(params)
    ax2.text(np.percentile(ts, 15), y2 + 0.2*(y2-y1), '\n'.join(text_left), va='bottom', ha='center')
    ax2.text(np.percentile(ts, 85), y2 + 0.2*(y2-y1), '\n'.join(text_right), va='bottom', ha='center')
    ax2.set_ylim([y1, y2])
    suffix = get_suffix(text_left, text_right) + '_tend' + str(t_end)
    plt.savefig(os.path.join(folder_name, 'dynamics' + suffix + '.pdf'), bbox_inches='tight')


if __name__ == "__main__":


    base_params = {
        'r_s': 0.03,      # Increased S growth rate
        'r_r': 0.01,      
        'c': 0.5,         
        'd': 1.5,         
        'K': 10000.0,
        't_s': 0.0001,    # Lowered transition slightly
        't_r': 0.0001,    
        'd_s': 0.0,
        'd_r': 0.0,
        's0': 1000.0,     # <--- Start mostly sensitive
        'r0': 1000.0,       # <--- Start with very few resistant cells!
        'dt': 1.0,
        't_end': 4000,    
        'folder_name': 'presentation_plots' 
    }

    # --- THERAPY SCHEDULES ---
    # Adaptive Therapy (Cytotoxic)
    at_schedule = {
        'r_s_treatment': 0.03,               
        'd_s_treatment': 0.06,               # Kills sensitive cells
        'threshold_treatment_on': 0.50,      # <--- Turn on at 50% capacity
        'threshold_treatment_off': 0.25,     # <--- Turn off at 25% capacity
    }

    # Constant Dose Therapy (Cytotoxic)
    cdt_schedule = {
        'r_s_treatment': 0.03,
        'd_s_treatment': 0.06,
        'threshold_treatment_on': 0.0,       
        'threshold_treatment_off': np.nan,  # Never turns off
    }
# =====================================================================
# PLOT 1: The Baseline (Original Model)
# =====================================================================
print("1. Generating Plot 1: Baseline (No Drug-Induced Plasticity)...")
plot_time_series(**base_params, **at_schedule, delta=0.0)
# =====================================================================
# PLOT 2: The "Moderate" Case
# =====================================================================
print("2. Generating Plot 2: Moderate Drug-Induced Plasticity...")
plot_time_series(**base_params, **at_schedule, delta=0.005)
# =====================================================================
# PLOT 3: The "Worst-Case" Extreme
# =====================================================================
print("3. Generating Plot 3: Extreme Drug-Induced Plasticity...")
plot_time_series(**base_params, **at_schedule, delta=0.05)
# =====================================================================
# PLOT 4: Why Constant Dose Fails
# =====================================================================
print("4. Generating Plot 4: Constant Dose Therapy (Moderate Plasticity)...")
plot_time_series(**base_params, **cdt_schedule, delta=0.005)
# =====================================================================
# PLOT 5: Drug-Induced Only (AC Model)
# =====================================================================
print("5. Generating Plot 5: AC Model (Drug-Induced Plasticity ONLY)...")
ac_params = base_params.copy()
ac_params['t_s'] = 0.0  # Zero out baseline plasticity
ac_params['t_r'] = 0.0  # Zero out baseline plasticity
plot_time_series(**ac_params, **at_schedule, delta=0.005)
print("==================================================")




















    # plot_time_series(
    # r_s=0.05, r_s_treatment=0.05, r_r=0.05,      # Cytotoxic mode: growth unaffected
    # c=0.5, d=0.5, K=10000,
    # t_s=0.0, delta=0.0, t_r=0.0,                 # STRICT AC MODEL: Zero plasticity
    # d_s=0.0, d_s_treatment=0.05, d_r=0.0,        # Cytotoxic death rate
    # threshold_treatment_on=0.5, threshold_treatment_off=0.25,
    # s0=4000, r0=500,                             # Start high to trigger treatment quickly
    # dt=0.1, t_end=2500,
    # )


    # plot_time_series(
    # r_s=0.05, r_s_treatment=0.05, r_r=0.05,
    # c=0.5, d=0.5, K=10000,
    # t_s=0.0, delta=0.005, t_r=0.0,               # DIP ACTIVATED: 0.005 added during treatment
    # d_s=0.0, d_s_treatment=0.05, d_r=0.0,
    # threshold_treatment_on=0.5, threshold_treatment_off=0.25,
    # s0=4000, r0=500,
    # dt=0.1, t_end=2500,
    # )

    # ------------------------------------------------------------------
    # delta = 0 throughout: recovers original model behaviour exactly.
    # ------------------------------------------------------------------

    # No Therapy Condition
    # never turns on (S+R will never exceed 1e10·K)
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01, t_s=0.0005, t_r=0.0005, delta=0., d_s=0, d_s_treatment=0., d_r=0, threshold_treatment_on=1e10,   
    # threshold_treatment_off=0.1,
    # s0=1000, r0=1000, t_end=400)   

    # Constant Dose Therapy (CDT): on immediately, never turns off
    # ts = tr = 0.0005
    # Cytostatic Therapy
    # no DIP (delta=0) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # no DIP (delta=0) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.0005) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.01) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.05) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.0005) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.01) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.05) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    



    # # Cytotoxic Therapy
    # # no DIP (delta=0) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0.01, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # no DIP (delta=0) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0.025, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # no DIP (delta=0) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0.1, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0.01, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0.01, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0.01, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0.025, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0.025, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0.025, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0.1, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0.1, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0.1, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    



    # # ts = tr = 0.01
    # # Cytostatic Therapy
    # # no DIP (delta=0) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # no DIP (delta=0) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.0005) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.01) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.05) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.0005) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.01) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta = 0.05) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0., threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    



    # # Cytotoxic Therapy
    # # no DIP (delta=0) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0.01, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # no DIP (delta=0) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0.025, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # no DIP (delta=0) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0.1, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0.01, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0.01, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0.01, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0.025, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0.025, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0.025, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0.1, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0.1, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0.1, threshold_treatment_on=0.0, threshold_treatment_off=np.nan, t_end=600)



    # Adaptive Therapy: on when S+R > 0.5·K, off when S+R < 0.1·K
    # ts = tr = 0.0005
    # Cytostatic Therapy
    # no DIP (delta=0) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0., t_end=600)
    
    # # no DIP (delta=0) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.0005) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.01) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.05) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.0005) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.01) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.05) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0., t_end=600)




    # # Cytotoxic Therapy
    # # no DIP (delta=0) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0.01, t_end=600)
    
    # # no DIP (delta=0) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0.025, t_end=600)
    
    # # no DIP (delta=0) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.,
    #                  d_s_treatment=0.1, t_end=1000)
    
    # # DIP (delta=0.0005) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0.01, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0.01, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0.01, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0.025, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0.025, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0.025, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.0005,
    #                  d_s_treatment=0.1, t_end=1000)
    
    # # DIP (delta=0.01) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.01,
    #                  d_s_treatment=0.1, t_end=1000)
    
    # # DIP (delta=0.05) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.0005, t_r=0.0005, delta=0.05,
    #                  d_s_treatment=0.1, t_end=1000)
    



    # ts = tr = 0.01
    # Cytostatic Therapy
    # no DIP (delta=0) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0., t_end=600)
    
    # # no DIP (delta=0) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.0005) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.01) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.05) & epsilon = 0.33
    # plot_time_series(r_s=0.01, r_s_treatment=0.0033, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.0005) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.01) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0., t_end=600)
    
    # # DIP (delta = 0.05) & epsilon = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.001, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0., t_end=600)
    



    # # Cytotoxic Therapy
    # # no DIP (delta=0) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0.01, t_end=600)
    
    # # no DIP (delta=0) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0.025, t_end=600)
    
    # # no DIP (delta=0) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.,
    #                  d_s_treatment=0.1, t_end=1000)
    
    # # DIP (delta=0.0005) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0.01, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0.01, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.01
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0.01, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0.025, t_end=600)
    
    # # DIP (delta=0.01) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0.025, t_end=600)
    
    # # DIP (delta=0.05) & dth = 0.025
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0.025, t_end=600)
    
    # # DIP (delta=0.0005) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.0005,
    #                  d_s_treatment=0.1, t_end=1000)
    
    # # DIP (delta=0.01) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.01,
    #                  d_s_treatment=0.1, t_end=1000)
    
    # # DIP (delta=0.05) & dth = 0.1
    # plot_time_series(r_s=0.01, r_s_treatment=0.01, r_r=0.01,
    #                  t_s=0.01, t_r=0.01, delta=0.05,
    #                  d_s_treatment=0.1, t_end=1000)




    # TIme to Progression (TTP) vs Delta Plots
    # delta_array = np.linspace(0.000125, 0.05, 1000)
    
    # Define the base parameters for the comparison (Cytostatic Example: epsilon = 0.33)
    # base_parameters = {
    #     'r_s': 0.01, 
    #     'r_s_treatment': 0.0033,  # epsilon * r_s
    #     'r_r': 0.01,
    #     'c': 0.4, 
    #     'd': 2.0, 
    #     'K': 10000,
    #     't_s': 0.0005, 
    #     't_r': 0.0005,
    #     'd_s': 0,
    #     'd_s_treatment': 0.0,     # cytostatic, so no added death rate
    #     'd_r': 0,
    #     'threshold_treatment_on': 0.5, 
    #     'threshold_treatment_off': 0.1,
    #     's0': 1000, 
    #     'r0': 1000, 
    #     'dt': 1.0
    # }

    # # You might need to increase t_end significantly (e.g., 5000 or 10000) 
    # # to ensure the tumor actually hits the 0.8K threshold, especially for small delta
    # plot_ttp_vs_delta(
    #     delta_values=delta_array, 
    #     progression_fraction=0.8, 
    #     t_end=5000, 
    #     **base_parameters)


    # base_parameters = {
    #     'r_s': 0.01, 
    #     'r_s_treatment': 0.001,  # epsilon * r_s
    #     'r_r': 0.01,
    #     'c': 2.0, 
    #     'd': 2.0, 
    #     'K': 10000,
    #     't_s': 0.0005, 
    #     't_r': 0.0005,
    #     'd_s': 0,
    #     'd_s_treatment': 0.0,     # cytostatic, so no added death rate
    #     'd_r': 0,
    #     'threshold_treatment_on': 0.5, 
    #     'threshold_treatment_off': 0.1,
    #     's0': 1000, 
    #     'r0': 1000, 
    #     'dt': 1.0
    # }

    # # You might need to increase t_end significantly (e.g., 5000 or 10000) 
    # # to ensure the tumor actually hits the 0.8K threshold, especially for small delta
    # plot_ttp_vs_delta(
    #     delta_values=delta_array, 
    #     progression_fraction=0.8, 
    #     t_end=5000, 
    #     **base_parameters
    # )


    # base_parameters = {
    #     'r_s': 0.01, 
    #     'r_s_treatment': 0.01,
    #     'r_r': 0.01,
    #     'c': 2.0, 
    #     'd': 2.0, 
    #     'K': 10000,
    #     't_s': 0.0005, 
    #     't_r': 0.0005,
    #     'd_s': 0,
    #     'd_s_treatment': 0.01,     # cytotoxic
    #     'd_r': 0,
    #     'threshold_treatment_on': 0.5, 
    #     'threshold_treatment_off': 0.1,
    #     's0': 1000, 
    #     'r0': 1000, 
    #     'dt': 1.0
    # }

    # # # You might need to increase t_end significantly (e.g., 5000 or 10000) 
    # # # to ensure the tumor actually hits the 0.8K threshold, especially for small delta
    # plot_ttp_vs_delta(
    #     delta_values=delta_array, 
    #     progression_fraction=0.8, 
    #     t_end=5000, 
    #     **base_parameters
    # )

    # base_parameters = {
    #     'r_s': 0.01, 
    #     'r_s_treatment': 0.01,  # epsilon * r_s
    #     'r_r': 0.01,
    #     'c': 2.0, 
    #     'd': 2.0, 
    #     'K': 10000,
    #     't_s': 0.0005, 
    #     't_r': 0.0005,
    #     'd_s': 0,
    #     'd_s_treatment': 0.025,     # cytotoxic
    #     'd_r': 0,
    #     'threshold_treatment_on': 0.5, 
    #     'threshold_treatment_off': 0.1,
    #     's0': 1000, 
    #     'r0': 1000, 
    #     'dt': 1.0
    # }

    # # # You might need to increase t_end significantly (e.g., 5000 or 10000) 
    # # # to ensure the tumor actually hits the 0.8K threshold, especially for small delta
    # plot_ttp_vs_delta(
    #     delta_values=delta_array, 
    #     progression_fraction=0.8, 
    #     t_end=5000, 
    #     **base_parameters
    # )

    # base_parameters = {
    #     'r_s': 0.01, 
    #     'r_s_treatment': 0.01,
    #     'r_r': 0.01,
    #     'c': 2.0, 
    #     'd': 2.0, 
    #     'K': 10000,
    #     't_s': 0.0005, 
    #     't_r': 0.0005,
    #     'd_s': 0,
    #     'd_s_treatment': 0.1,     # cytotoxic
    #     'd_r': 0,
    #     'threshold_treatment_on': 0.5, 
    #     'threshold_treatment_off': 0.1,
    #     's0': 1000, 
    #     'r0': 1000, 
    #     'dt': 1.0
    # }

    # # You might need to increase t_end significantly (e.g., 5000 or 10000) 
    # # to ensure the tumor actually hits the 0.8K threshold, especially for small delta
    # plot_ttp_vs_delta(
    #     delta_values=delta_array, 
    #     progression_fraction=0.8, 
    #     t_end=5000, 
    #     **base_parameters
    # )