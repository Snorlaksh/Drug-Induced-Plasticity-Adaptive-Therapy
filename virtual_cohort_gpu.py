"""
GPU-accelerated Virtual Cohort Simulation with DIP
==================================================
Reads the exact parameters and outcomes from the Constant Dose Therapy simulation
and re-runs those exact virtual patients under Adaptive Therapy for direct comparison.
Fixes applied:
  1. t_delay now correctly implements a pre-therapy free-growth phase (Phase 1)
     followed by adaptive therapy (Phase 2). FFT runs on Phase 2 only.
  2. delta validated and clipped to [0, DELTA_MAX] before GPU upload.
  3. BATCH_SIZE properly used to avoid GPU out-of-memory.
  4. AT switching logic simplified (redundant else removed).
  5. RK4 midpoint clamping retained throughout.
"""
import numpy as np
import pandas as pd
from numba import cuda
import time
import os
import math
# --- 1. SETTINGS ---
BATCH_SIZE    = 40000
THERAPY_STEPS = 4000       # steps of adaptive therapy (Phase 2)
K_CAPACITY    = 10000.0
DELTA_MAX     = 0.05       # enforced upper bound for drug-induced plasticity
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis')
OUT_DIR  = os.path.join(DATA_DIR, 'adaptive-therapy')
os.makedirs(OUT_DIR, exist_ok=True)
# --- 2. CUDA KERNEL ---
@cuda.jit
def simulate_adaptive_kernel(params, out_states, out_timeseries):
    """
    Two-phase simulation per virtual patient:
      Phase 1 (steps 0 … t_delay-1) : free growth, no therapy.
      Phase 2 (steps t_delay … end)  : adaptive therapy with hysteresis.
    Only Phase 2 population totals are written to out_timeseries (for FFT).
    params columns:
      0: g_s      – sensitive growth rate  (r_x)
      1: g_r      – resistant growth rate  (r_y)
      2: a        – transition rate S→R    (t_s)
      3: b        – transition rate R→S    (t_r)
      4: c        – competition coeff α_rs (effect of R on S)
      5: d        – competition coeff α_sr (effect of S on R)
      6: delta    – drug-induced plasticity (active during therapy only)
      7: t_delay  – pre-therapy free-growth steps (e.g. 1000)
      8: g_scale  – cytostatic divisor (1.0 = none; >1 = cytostatic active)
      9: d_th     – cytotoxic death rate  (0.0 = none)
    """
    idx = cuda.grid(1)
    if idx >= params.shape[0]:
        return
    g_s     = params[idx, 0]
    g_r     = params[idx, 1]
    a       = params[idx, 2]
    b       = params[idx, 3]
    c       = params[idx, 4]
    d       = params[idx, 5]
    delta   = params[idx, 6]
    t_delay = params[idx, 7]   # float threshold; compared to step counter
    g_scale = params[idx, 8]
    d_th    = params[idx, 9]
    S  = 50.0
    R  = 50.0
    dt = 1.0
    total_steps = int(t_delay) + THERAPY_STEPS
    t_on   = False
    ts_idx = 0     # write index into out_timeseries (Phase 2 only)
    for step in range(total_steps):
        S_curr = S
        R_curr = R
        in_therapy_phase = (step >= t_delay)
        # ── Therapy on/off logic ────────────────────────────────────────────
        if in_therapy_phase:
            total_curr = S_curr + R_curr
            # Hysteresis: on above 50% K, off below 10% K, unchanged in between
            if total_curr > 0.5 * K_CAPACITY:
                t_on = True
            elif t_on and total_curr < 0.1 * K_CAPACITY:
                t_on = False
            # else: t_on unchanged
        else:
            t_on = False   # Phase 1: no therapy
        # ── Effective parameters ────────────────────────────────────────────
        if t_on:
            g_s_eff  = g_s / g_scale
            d_th_eff = d_th
            # delta only active when actual drug present (cytostatic or cytotoxic)
            if g_scale > 1.0 or d_th > 0.0:
                a_eff = a + delta
            else:
                a_eff = a
        else:
            g_s_eff  = g_s
            d_th_eff = 0.0
            a_eff    = a
        # ── RK4 integration ─────────────────────────────────────────────────
        # k1
        dS_k1 = (g_s_eff * S_curr * (K_CAPACITY - (S_curr + c * R_curr)) / K_CAPACITY
                 - a_eff * S_curr + b * R_curr - d_th_eff * S_curr)
        dR_k1 = (g_r * R_curr * (K_CAPACITY - (d * S_curr + R_curr)) / K_CAPACITY
                 + a_eff * S_curr - b * R_curr)
        # k2
        S_k2 = S_curr + 0.5 * dt * dS_k1
        R_k2 = R_curr + 0.5 * dt * dR_k1
        if S_k2 < 0.0: S_k2 = 0.0
        if R_k2 < 0.0: R_k2 = 0.0
        dS_k2 = (g_s_eff * S_k2 * (K_CAPACITY - (S_k2 + c * R_k2)) / K_CAPACITY
                 - a_eff * S_k2 + b * R_k2 - d_th_eff * S_k2)
        dR_k2 = (g_r * R_k2 * (K_CAPACITY - (d * S_k2 + R_k2)) / K_CAPACITY
                 + a_eff * S_k2 - b * R_k2)
        # k3
        S_k3 = S_curr + 0.5 * dt * dS_k2
        R_k3 = R_curr + 0.5 * dt * dR_k2
        if S_k3 < 0.0: S_k3 = 0.0
        if R_k3 < 0.0: R_k3 = 0.0
        dS_k3 = (g_s_eff * S_k3 * (K_CAPACITY - (S_k3 + c * R_k3)) / K_CAPACITY
                 - a_eff * S_k3 + b * R_k3 - d_th_eff * S_k3)
        dR_k3 = (g_r * R_k3 * (K_CAPACITY - (d * S_k3 + R_k3)) / K_CAPACITY
                 + a_eff * S_k3 - b * R_k3)
        # k4
        S_k4 = S_curr + dt * dS_k3
        R_k4 = R_curr + dt * dR_k3
        if S_k4 < 0.0: S_k4 = 0.0
        if R_k4 < 0.0: R_k4 = 0.0
        dS_k4 = (g_s_eff * S_k4 * (K_CAPACITY - (S_k4 + c * R_k4)) / K_CAPACITY
                 - a_eff * S_k4 + b * R_k4 - d_th_eff * S_k4)
        dR_k4 = (g_r * R_k4 * (K_CAPACITY - (d * S_k4 + R_k4)) / K_CAPACITY
                 + a_eff * S_k4 - b * R_k4)
        # Final RK4 update
        S = S_curr + (dt / 6.0) * (dS_k1 + 2.0 * dS_k2 + 2.0 * dS_k3 + dS_k4)
        R = R_curr + (dt / 6.0) * (dR_k1 + 2.0 * dR_k2 + 2.0 * dR_k3 + dR_k4)
        if S < 0.0: S = 0.0
        if R < 0.0: R = 0.0
        # ── Record timeseries for FFT (Phase 2 / therapy phase only) ────────
        if in_therapy_phase:
            out_timeseries[idx, ts_idx] = S + R
            ts_idx += 1
    # Final state
    out_states[idx, 0] = S
    out_states[idx, 1] = R
# --- 3. FFT PERIOD DETECTION (CPU VECTORISED) ---
def detect_periods_batch(timeseries, dt=1.0):
    """
    Batch period detection using FFT.
    Returns np.inf for non-cycling (steady-state or no dominant spectral peak).
    """
    n_sims, n_samples = timeseries.shape
    means     = timeseries.mean(axis=1, keepdims=True)
    detrended = timeseries - means
    fft_result = np.fft.rfft(detrended, axis=1)
    # Exclude DC component (index 0)
    power = np.abs(fft_result[:, 1:]) ** 2
    freqs = np.fft.rfftfreq(n_samples, d=dt)[1:]
    dominant_idx   = np.argmax(power, axis=1)
    dominant_freq  = freqs[dominant_idx]
    dominant_power = power[np.arange(n_sims), dominant_idx]
    periods = np.where(dominant_freq > 0, 1.0 / dominant_freq, np.inf)
    # Noise guard: dominant peak must carry >= 5% of total spectral power
    total_power = power.sum(axis=1)
    frac_power  = np.where(total_power > 0, dominant_power / total_power, 0.0)
    periods     = np.where(frac_power >= 0.05, periods, np.inf)
    
    # ── NEW FIX: Filter out slow-convergence artifacts ──
    MAX_GENUINE_PERIOD = 1000.0
    periods = np.where(periods < MAX_GENUINE_PERIOD, periods, np.inf)
    
    return periods
# --- 4. BATCHED GPU RUNNER ---
def run_batched(params_all, batch_size=BATCH_SIZE):
    """
    Runs the adaptive kernel in batches to avoid GPU OOM.
    Returns:
        all_states  : np.ndarray [n_sims, 2] — final (S, R)
        all_periods : np.ndarray [n_sims]    — oscillation period or np.inf
    """
    n_sims      = params_all.shape[0]
    all_states  = np.zeros((n_sims, 2), dtype=np.float32)
    all_periods = np.full(n_sims, np.inf, dtype=np.float64)
    n_batches = math.ceil(n_sims / batch_size)
    for batch_idx, batch_start in enumerate(range(0, n_sims, batch_size)):
        batch_end    = min(batch_start + batch_size, n_sims)
        n_batch      = batch_end - batch_start
        batch_params = np.ascontiguousarray(
            params_all[batch_start:batch_end].astype(np.float32)
        )
        out_states_b = np.zeros((n_batch, 2),             dtype=np.float32)
        out_ts_b     = np.zeros((n_batch, THERAPY_STEPS), dtype=np.float32)
        threads_per_block = 256
        blocks_per_grid   = math.ceil(n_batch / threads_per_block)
        d_params = cuda.to_device(batch_params)
        d_states = cuda.to_device(out_states_b)
        d_ts     = cuda.to_device(out_ts_b)
        simulate_adaptive_kernel[blocks_per_grid, threads_per_block](
            d_params, d_states, d_ts
        )
        cuda.synchronize()
        d_states.copy_to_host(out_states_b)
        d_ts.copy_to_host(out_ts_b)
        all_states[batch_start:batch_end]  = out_states_b
        
        # ── NEW FIX: Physical check to discard cytostatic pseudo-cycling ──
        min_pop = out_ts_b.min(axis=1)
        never_cycled = min_pop > 1000.0  # 10% of K_CAPACITY (10,000)
        periods_b = detect_periods_batch(out_ts_b)
        periods_b[never_cycled] = np.inf
        
        all_periods[batch_start:batch_end] = periods_b

        print(f"    Batch {batch_idx + 1}/{n_batches} "
              f"({batch_start}–{batch_end}) done.")
    return all_states, all_periods
# --- 5. EXECUTION ---
if __name__ == '__main__':
    print("=================================================================")
    print("  Virtual Cohort Simulation — Constant vs Adaptive Therapy")
    print("=================================================================\n")
    if not cuda.is_available():
        print("  ERROR: No CUDA-capable GPU detected.")
        exit(1)
    FILES_TO_PROCESS = {
        'no-therapy':      'no-therapy-outcomes-r3.csv',
        'cytostatic-0.33': 'CDT-cytostatic-0.33-zero-delay-r3.csv',
        'cytostatic-0.1':  'CDT-cytostatic-0.1-zero-delay-r3.csv',
        'cytotoxic-0.01':  'CDT-cytotoxic-0.01-zero-delay-r3.csv',
        'cytotoxic-0.05':  'CDT-cytotoxic-0.05-zero-delay-r3.csv',
    }
    t_total_start = time.time()
    for therapy_name, filename in FILES_TO_PROCESS.items():
        print(f"--- Processing: {therapy_name} ---")
        cdt_file_path = os.path.join(DATA_DIR, 'constant-dose', filename)
        if not os.path.exists(cdt_file_path):
            print(f"  WARNING: File not found ({cdt_file_path}). Skipping.\n")
            continue
        # ── Load CDT data ─────────────────────────────────────────────────────
                # ── Load CDT data ─────────────────────────────────────────────────────
        df_cdt = pd.read_csv(cdt_file_path)
        
        # ── NEW FIX: Load no-therapy baseline outcomes for perfect row-alignment ──
        notherapy_path = os.path.join(DATA_DIR, 'constant-dose', 'no-therapy-outcomes-r3.csv')
        df_notherapy = pd.read_csv(notherapy_path)
        df_cdt['ResFracBT'] = df_notherapy['ResFraction'].values
        df_cdt['PopSizeBT'] = df_notherapy['PopSizeSS'].values
        
        # ── NEW FIX: Implement Phase 1 (delayed therapy) before sampling ──
        df_cdt['t_delay'] = 1000.0
        
        # ── Validate and clip delta to [0, DELTA_MAX] ─────────────────────────
        raw_max = df_cdt['delta'].max()
        if raw_max > DELTA_MAX:
            print(f"  WARNING: delta exceeds {DELTA_MAX} "
                  f"(max found = {raw_max:.5f}). Clipping.")
            df_cdt['delta'] = df_cdt['delta'].clip(lower=0.0, upper=DELTA_MAX)
        # ── ModelType classification ───────────────────────────────────────────
        df_cdt['ModelType'] = 'Unknown'
        df_cdt.loc[(df_cdt['c'] != df_cdt['d']) & (df_cdt['a'] > 0), 'ModelType'] = 'AC_Tr'
        df_cdt.loc[(df_cdt['c'] != df_cdt['d']) & (df_cdt['a'] == 0), 'ModelType'] = 'AC_No_Tr'
        df_cdt.loc[(df_cdt['c'] == df_cdt['d']) & (df_cdt['a'] > 0), 'ModelType'] = 'SC_Tr'
        df_cdt.loc[(df_cdt['c'] == df_cdt['d']) & (df_cdt['a'] < 1e-12), 'ModelType'] = 'SC_No_Tr'
        # ── Balanced sampling (up to 10 000 per model type) ───────────────────
        sampled_dfs = []
        for mtype in ['AC_Tr', 'AC_No_Tr', 'SC_Tr', 'SC_No_Tr']:
            subset   = df_cdt[df_cdt['ModelType'] == mtype]
            n_sample = min(10_000, len(subset))
            sampled_dfs.append(subset.sample(n=n_sample, random_state=42))
        cohort_df = pd.concat(sampled_dfs).reset_index(drop=True)
        n_sims    = len(cohort_df)
        print(f"  Cohort size: {n_sims:,} virtual patients")
        # ── Build params array (column order must match kernel indices 0–9) ───
        params = cohort_df[[
            'r_x', 'r_y',    # 0, 1
            'a', 'b',         # 2, 3
            'c', 'd',         # 4, 5
            'delta',          # 6
            't_delay',        # 7
            'g_scale',        # 8
            'cell_death',     # 9
        ]].to_numpy(dtype=np.float64)
        # ── Run in batches ────────────────────────────────────────────────────
        out_states, periods = run_batched(params, batch_size=BATCH_SIZE)
        # ── Compute AT outcomes ───────────────────────────────────────────────
        sen_at = out_states[:, 0]
        res_at = out_states[:, 1]
        pop_at = sen_at + res_at
        res_frac_at       = np.zeros_like(res_at)
        mask              = pop_at > 0
        res_frac_at[mask] = res_at[mask] / pop_at[mask]
        # ── Assemble output dataframe ─────────────────────────────────────────
        cohort_df = cohort_df.rename(columns={
            'ResFraction': 'ResFracCDT',
            'PopSizeSS':   'PopSizeCDT',
        })
        cohort_df['ResFracAT'] = res_frac_at
        cohort_df['PopSizeAT'] = pop_at
        cohort_df['PeriodAT']  = periods
        cohort_df.drop(columns=['sen', 'res', 'ModelType'],
                       inplace=True, errors='ignore')
        out_path = os.path.join(OUT_DIR, f'virtual-cohort-{therapy_name}-r3.csv')
        cohort_df.to_csv(out_path, index=False)
        # ── Summary ───────────────────────────────────────────────────────────
        cycling = int(np.sum(np.isfinite(periods) & (periods < THERAPY_STEPS)))
        unfav   = int(np.sum(np.isinf(periods)    & (res_frac_at > 0.15)))
        fav     = int(np.sum(np.isinf(periods)    & (res_frac_at <= 0.15)))
        print(f"  Saved  → {out_path}")
        print(f"  AT Outcomes: Cycling={cycling:,}  "
              f"Unfavourable={unfav:,}  Favourable={fav:,}\n")
    print("=================================================================")
    print(f"  ALL DONE. Total time: {time.time() - t_total_start:.1f}s")
    print("=================================================================")
