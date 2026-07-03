"""
GPU-accelerated Adaptive Therapy Simulation with Drug-Induced Plasticity (DIP)
===============================================================================
Target hardware: NVIDIA RTX 3050 (4 GB VRAM) + Intel i5-12500H

Requires:
    pip install numba scipy pandas numpy

    + NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
    + Compatible NVIDIA driver

Usage:
    python adaptive_therapy_gpu.py

Output:
    5 CSV files in ./analysis/adaptive-therapy/
"""

import numpy as np
from numba import cuda, float32 as f32
from scipy import stats
import pandas as pd
import time
import os
import sys

# =====================================================================
# Simulation Constants
# =====================================================================
K_VAL       = 10000.0
DT_VAL      = 1.0
N_STEPS     = 5000       # t_max / dt
S0_VAL      = 50.0
R0_VAL      = 50.0
THR_ON      = 0.5        # Therapy ON  when total > THR_ON  * K
THR_OFF     = 0.1        # Therapy OFF when total < THR_OFF * K

# GPU tuning
THREADS_PER_BLOCK = 256

# Output directory (relative to script location)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'analysis', 'adaptive-therapy')


# =====================================================================
# CUDA Kernel — one thread per simulation
# =====================================================================
@cuda.jit
def simulate_kernel(params, total_pop, y_final):
    """
    Each CUDA thread runs one full RK4 adaptive therapy simulation.

    params[i, 0:10] = [r_x, r_y, a, b, c, d, delta, t_delay, g_scale, d_th]
    total_pop[i, 0:5001]  — total population at each time step (float32)
    y_final[i, 0:2]       — [s_final, r_final] (float32)
    """
    idx = cuda.grid(1)
    if idx >= params.shape[0]:
        return

    # --- Extract parameters ---
    r_x     = params[idx, 0]
    r_y     = params[idx, 1]
    a       = params[idx, 2]   # t_s: basal s->r transition rate
    b       = params[idx, 3]   # t_r: r->s transition rate
    c_comp  = params[idx, 4]   # competition: R on S
    d_comp  = params[idx, 5]   # competition: S on R
    delta   = params[idx, 6]   # DIP uplift
    t_delay = params[idx, 7]   # therapy delay
    g_scale = params[idx, 8]   # cytostatic scaling
    d_th    = params[idx, 9]   # cytotoxic death rate

    # Derived therapy parameters
    r_x_th = r_x / g_scale     # growth rate under cytostatic therapy

    
    # Local constants (float32 for GPU perf)
    Kv   = f32(10000.0)
    dt   = f32(1.0)
    half = f32(0.5)
    sixth = f32(1.0) / f32(6.0)
    two  = f32(2.0)
    one  = f32(1.0)
    zero = f32(0.0)
    thr_on_K  = f32(0.5) * Kv   # 5000
    thr_off_K = f32(0.1) * Kv   # 1000

    # Initial state
    s = f32(50.0)
    rc = f32(50.0)             # r_cell (avoid shadowing r_y)
    t_on = False

    total_pop[idx, 0] = s + rc

    # Before the loop:
    has_therapy = (g_scale > 1.0) or (d_th > 0.0)
    a_th = a + delta if has_therapy else a   # DIP only if drug is present 

    for step in range(5000):
        # Select effective parameters based on therapy state
        if t_on:
            rs = r_x_th
            ae = a_th
            de = d_th
        else:
            rs = r_x
            ae = a
            de = zero

        # ---- RK4 Step ----
        # k1
        ds1 = rs * s  * (one - (s  + c_comp * rc) / Kv) - ae * s  + b * rc - de * s
        dr1 = r_y * rc * (one - (rc + d_comp * s)  / Kv) + ae * s  - b * rc

        # k2
        sh = s  + half * dt * ds1
        rh = rc + half * dt * dr1
        if sh < zero: sh = zero
        if rh < zero: rh = zero

        ds2 = rs * sh * (one - (sh + c_comp * rh) / Kv) - ae * sh + b * rh - de * sh
        dr2 = r_y * rh * (one - (rh + d_comp * sh) / Kv) + ae * sh - b * rh

        # k3
        sh = s  + half * dt * ds2
        rh = rc + half * dt * dr2
        if sh < zero: sh = zero
        if rh < zero: rh = zero

        ds3 = rs * sh * (one - (sh + c_comp * rh) / Kv) - ae * sh + b * rh - de * sh
        dr3 = r_y * rh * (one - (rh + d_comp * sh) / Kv) + ae * sh - b * rh

        # k4
        sh = s  + dt * ds3
        rh = rc + dt * dr3
        if sh < zero: sh = zero
        if rh < zero: rh = zero

        ds4 = rs * sh * (one - (sh + c_comp * rh) / Kv) - ae * sh + b * rh - de * sh
        dr4 = r_y * rh * (one - (rh + d_comp * sh) / Kv) + ae * sh - b * rh

        # Update
        s  = s  + sixth * dt * (ds1 + two * ds2 + two * ds3 + ds4)
        rc = rc + sixth * dt * (dr1 + two * dr2 + two * dr3 + dr4)

        # Clamp negatives
        if s  < zero:
            s  = zero
        if rc < zero:
            rc = zero

        total = s + rc

        # Adaptive therapy switching (hysteresis)
        if not t_on and total > thr_on_K:
            t_on = True
        elif t_on and total < thr_off_K:
            t_on = False

        # Store total population for period detection
        total_pop[idx, step + 1] = total

    # Store final state
    y_final[idx, 0] = s
    y_final[idx, 1] = rc


# =====================================================================
# Period Detection — vectorized FFT on CPU
# =====================================================================
def detect_periods_batch(total_pop_array):
    """
    Batch period detection using numpy FFT.

    Parameters
    ----------
    total_pop_array : ndarray (N, 5001) — total population time series

    Returns
    -------
    periods : ndarray (N,) — dominant oscillation period (inf if none)
    """
    n_sims, n_points = total_pop_array.shape

    # Detrend: subtract mean per simulation
    means = total_pop_array.mean(axis=1, keepdims=True)
    detrended = total_pop_array - means

    # FFT (real input)
    fft_result = np.fft.rfft(detrended, axis=1)
    power = np.abs(fft_result[:, 1:]) ** 2  # skip DC component

    # Frequency axis
    freqs = np.fft.rfftfreq(n_points, d=DT_VAL)[1:]  # skip DC

    # Find dominant frequency per simulation
    dominant_idx = np.argmax(power, axis=1)
    dominant_power = power[np.arange(n_sims), dominant_idx]

    # Periods
    dominant_freq = freqs[dominant_idx]
    periods = np.where(dominant_freq > 0, 1.0 / dominant_freq, np.inf)

    # If dominant power is negligible, no meaningful oscillation
    total_power = power.sum(axis=1)
    frac_power = np.where(total_power > 0,
                          dominant_power / total_power,
                          0.0)
    # If the dominant peak carries < 5% of spectral power, call it non-cycling
    periods = np.where(frac_power >= 0.05, periods, np.inf)

    MAX_GENUINE_PERIOD = 1000.0
    periods = np.where(periods < MAX_GENUINE_PERIOD, periods, np.inf)

    return periods


# =====================================================================
# GPU Batch Runner
# =====================================================================
def run_batch_gpu(params_batch):
    """
    Launch GPU kernel for a batch of simulations.

    Parameters
    ----------
    params_batch : ndarray (N, 10) float64

    Returns
    -------
    y_final    : ndarray (N, 2) — [s_ss, r_ss]
    periods    : ndarray (N,)   — oscillation period
    popsize_ss : ndarray (N,)   — total steady-state population
    """
    n = params_batch.shape[0]

    # Transfer to GPU as float32
    params_gpu    = cuda.to_device(params_batch.astype(np.float32))
    total_pop_gpu = cuda.device_array((n, N_STEPS + 1), dtype=np.float32)
    y_final_gpu   = cuda.device_array((n, 2), dtype=np.float32)

    # Launch kernel
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    simulate_kernel[blocks, THREADS_PER_BLOCK](params_gpu, total_pop_gpu, y_final_gpu)
    cuda.synchronize()

    # Transfer back to CPU
    total_pop = total_pop_gpu.copy_to_host().astype(np.float64)
    y_final   = y_final_gpu.copy_to_host().astype(np.float64)

    # Period detection (vectorized FFT on CPU)
    # Transfer back to CPU
    total_pop = total_pop_gpu.copy_to_host().astype(np.float64)
    y_final   = y_final_gpu.copy_to_host().astype(np.float64)

    # ── NEW FIX: Physical check to discard cytostatic pseudo-cycling ──
    # We slice [:, 2501:] to match the FFT window exactly
    min_pop = total_pop[:, 2501:].min(axis=1)
    never_cycled = min_pop > 1000.0  # 10% of K_CAPACITY (10,000)
    
    periods = detect_periods_batch(total_pop[:, 2501:])
    periods[never_cycled] = np.inf
    # ──────────────────────────────────────────────────────────────────

    popsize_ss = y_final[:, 0] + y_final[:, 1]

    # Free GPU memory
    del params_gpu, total_pop_gpu, y_final_gpu

    return y_final, periods, popsize_ss


# =====================================================================
# Parameter Generation (matches Rmd logic)
# =====================================================================
def generate_parameters():
    """
    Generate all parameter sets for 5 therapy conditions.

    Returns dict: {csv_filename: params_array (N, 10)}
    """
    # ---- LHS sampling (7 dims: r_x, r_y, a, b, c, d, delta) ----
    x = np.array([[0.5**i, 1] for i in np.arange(1, 4)])
    y = np.array([[1, 0.5**i] for i in np.arange(0, 4)])
    wg = np.append(x, y, axis=0).round(decimals=3)

    growth_rate_fast = (1/20) * wg
    growth_rate_slow = (1/50) * wg

    comp_strong = np.asarray([[2, 2*i] for i in np.arange(0.5, 2.1, 0.1)]).round(decimals=2)
    comp_weak   = np.asarray([[0.5, 0.5*i] for i in np.arange(0.5, 2.1, 0.1)]).round(decimals=2)

    transition_rate_fast = (1/20/10) * wg
    transition_rate_slow = (1/1000) * wg

    l_bound = [growth_rate_slow.min(), growth_rate_slow.min(),
               transition_rate_slow.min(), transition_rate_slow.min(),
               comp_weak.min(), comp_weak.min(),
               1e-4]   # delta lower
    u_bound = [growth_rate_fast.max(), growth_rate_fast.max(),
               transition_rate_fast.max(), transition_rate_fast.max(),
               comp_strong.max(), comp_strong.max(),
               0.05]       # delta upper

    sample = stats.qmc.LatinHypercube(d=7).random(n=5000)
    parms_latin = stats.qmc.scale(sample, l_bound, u_bound)

    # ---- Log-uniform sampling ----
    r_log     = np.exp(np.random.uniform(-6, -3, 10000)).reshape(5000, 2)
    tr_log    = np.exp(np.random.uniform(-8.5, -6.5, 10000)).reshape(5000, 2)
    comp_log  = np.exp(np.random.uniform(-2, 2, 10000)).reshape(5000, 2)
    delta_log = np.exp(np.random.uniform(
                    np.log(1e-4), np.log(0.05), 5000)).reshape(5000, 1)
    parms_log = np.hstack([r_log, tr_log, comp_log, delta_log])

    # Union
    parms_array = np.concatenate([parms_latin, parms_log])
    # columns: [r_x, r_y, a, b, c, d, delta]

    # ---- 4 model variants ----
    ac_tr = parms_array.copy()

    ac_no_tr = parms_array.copy()
    ac_no_tr[:, 2:4] = 0.0
    # delta (column 6) is NOT zeroed out: DIP is present even without basal transitions

    sc_tr = parms_array.copy()
    sc_tr[:, 5] = sc_tr[:, 4]  # Symmetric competition

    sc_no_tr = sc_tr.copy()
    sc_no_tr[:, 2:4] = 0.0
    # delta (column 6) is NOT zeroed out: DIP is present even without basal transitions

    base = np.concatenate([ac_tr, ac_no_tr, sc_tr, sc_no_tr])
    # columns: [r_x, r_y, a, b, c, d, delta]

    # Add t_delay = 0
    base = np.column_stack([base, np.zeros(len(base))])
    # columns: [r_x, r_y, a, b, c, d, delta, t_delay]

    n = len(base)

    # ---- 5 therapy conditions ----
    # Append [g_scale, d_th] → 10 columns total
    conditions = {
        'no-therapy-outcomes-r3':
            np.column_stack([base, np.ones(n),           np.zeros(n)]),
        'AT-cytostatic-0.33-zero-delay-r3':
            np.column_stack([base, np.full(n, 3.0),      np.zeros(n)]),
        'AT-cytostatic-0.1-zero-delay-r3':
            np.column_stack([base, np.full(n, 10.0),     np.zeros(n)]),
        'AT-cytotoxic-0.01-zero-delay-r3':
            np.column_stack([base, np.ones(n),           np.full(n, 0.01)]),
        'AT-cytotoxic-0.05-zero-delay-r3':
            np.column_stack([base, np.ones(n),           np.full(n, 0.05)]),
    }

    return conditions


# =====================================================================
# Main
# =====================================================================
def main():
    print('=' * 65)
    print('  Adaptive Therapy GPU Simulation — DIP Extension')
    print('  Target: NVIDIA RTX 3050 (4 GB VRAM)')
    print('=' * 65)

    # ---- Check GPU ----
    if not cuda.is_available():
        print('\n  ERROR: No CUDA-capable GPU detected.')
        print('  Make sure you have:')
        print('    1. NVIDIA driver installed')
        print('    2. CUDA Toolkit installed')
        print('    3. numba[cuda] installed (pip install numba)')
        sys.exit(1)

    device = cuda.get_current_device()
    print(f'\n  GPU : {device.name.decode()}')
    mem = cuda.current_context().get_memory_info()
    print(f'  VRAM: {mem[1] / 1e9:.1f} GB total, {mem[0] / 1e9:.1f} GB free')

    # ---- Generate parameters ----
    print('\n  Generating parameters...')
    t0 = time.time()
    all_conditions = generate_parameters()
    total_sims = sum(len(v) for v in all_conditions.values())
    print(f'  Total simulations: {total_sims:,}')
    print(f'  Parameter generation: {time.time() - t0:.1f}s')

    # ---- Create output directory ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'  Output directory: {OUTPUT_DIR}')

    # ---- Warmup (JIT compile the kernel) ----
    print('\n  Compiling CUDA kernel (first-time warmup)...')
    t0 = time.time()
    warmup_p = np.array([[0.01, 0.01, 0.001, 0.001,
                          1.0, 1.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    warmup_tp = cuda.device_array((1, N_STEPS + 1), dtype=np.float32)
    warmup_yf = cuda.device_array((1, 2), dtype=np.float32)
    simulate_kernel[1, 1](cuda.to_device(warmup_p), warmup_tp, warmup_yf)
    cuda.synchronize()
    del warmup_p, warmup_tp, warmup_yf
    print(f'  Kernel compiled in {time.time() - t0:.1f}s')

    # ---- Run all conditions ----
    print('\n  Starting simulations...')
    print('-' * 65)
    t_total = time.time()

    for name, params in all_conditions.items():
        n = len(params)
        print(f'\n  [{name}]')
        print(f'    Simulations : {n:,}')

        # Check VRAM requirement
        vram_mb = (n * (N_STEPS + 1) * 4 + n * 10 * 4 + n * 2 * 4) / 1e6
        print(f'    VRAM needed : {vram_mb:.0f} MB')

        t0 = time.time()

        # Run on GPU
        y_final, periods, popsize_ss = run_batch_gpu(params)
        t_gpu = time.time() - t0

        print(f'    GPU time    : {t_gpu:.1f}s')

        # Build DataFrame
        df = pd.DataFrame(params, columns=[
            'r_x', 'r_y', 'a', 'b', 'c', 'd', 'delta',
            't_delay', 'g_scale', 'cell_death'
        ])
        df['sen']         = y_final[:, 0]
        df['res']         = y_final[:, 1]
        total_cells       = df['sen'] + df['res']
        df['ResFraction'] = np.where(total_cells > 0,
                                     df['res'] / total_cells, np.nan)
        df['Period']      = periods
        df['PopSizeSS']   = popsize_ss

        # Save
        fpath = os.path.join(OUTPUT_DIR, f'{name}.csv')
        df.to_csv(fpath, index=False)
        print(f'    Saved       : {fpath}')

        # Quick summary
        n_cycling = np.sum(np.isfinite(periods) & (periods < N_STEPS))
        n_unfav   = np.sum(np.isinf(periods) & (df['ResFraction'] > 0.15))
        n_fav     = np.sum(np.isinf(periods) & (df['ResFraction'] <= 0.15))
        print(f'    Outcomes    : Cycling={n_cycling:,}  '
              f'Unfavourable={n_unfav:,}  Favourable={n_fav:,}')

    t_elapsed = time.time() - t_total
    print(f'\n{"=" * 65}')
    print(f'  DONE. Total time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)')
    print(f'  Output: {OUTPUT_DIR}')
    print(f'{"=" * 65}')


if __name__ == '__main__':
    main()
