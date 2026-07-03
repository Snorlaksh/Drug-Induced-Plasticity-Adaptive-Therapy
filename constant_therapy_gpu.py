"""
GPU-accelerated Constant Dose Therapy Simulation with Drug-Induced Plasticity (DIP)
=================================================================================
Target hardware: NVIDIA RTX 3050 (4 GB VRAM) + Intel i5-12500H
"""
import numpy as np
import pandas as pd
from numba import cuda
import time
import os
import math
from scipy import stats

# --- 1. SETTINGS & CONSTANTS ---
BATCH_SIZE = 40000 
K_CAPACITY = 10000.0

# Ensure output directory exists
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis', 'constant-dose')
os.makedirs(OUT_DIR, exist_ok=True)

# --- 2. CUDA KERNEL ---
@cuda.jit
def simulate_constant_therapy_kernel(params, out_states):
    """
    Each GPU thread runs one simulation of 5000 hours of constant therapy.
    """
    idx = cuda.grid(1)
    if idx >= params.shape[0]:
        return
        
    # Extract parameters
    g_s = params[idx, 0]
    g_r = params[idx, 1]
    a   = params[idx, 2]
    b   = params[idx, 3]
    c   = params[idx, 4]
    d   = params[idx, 5]
    delta = params[idx, 6]
    
    t_delay = params[idx, 7]
    g_scale = params[idx, 8]
    d_th    = params[idx, 9]
    
    # Determine if therapy is applied
    t_on = False
    if g_scale > 1.0 or d_th > 0.0:
        t_on = True
    
    # In constant therapy, the therapy is ALWAYS ON (or ALWAYS OFF if no_treat)
    a_eff = a + delta if t_on else a
    g_s_eff = g_s / g_scale
    
    S = 50.0
    R = 50.0
    dt = 1.0
    steps = 5000
    
    # RK4 Integration loop
    for step in range(steps):
        # Current states
        S_curr = S
        R_curr = R
        #total_curr = S_curr + R_curr
        
        # --- k1 ---
        dS_k1 = g_s_eff * S_curr * (K_CAPACITY - (S_curr + c * R_curr)) / K_CAPACITY - a_eff * S_curr + b * R_curr - d_th * S_curr
        dR_k1 = g_r     * R_curr * (K_CAPACITY - (d * S_curr + R_curr)) / K_CAPACITY + a_eff * S_curr - b * R_curr
        
        # --- k2 ---
        S_k2 = S_curr + 0.5 * dt * dS_k1
        R_k2 = R_curr + 0.5 * dt * dR_k1
        if S_k2 < 0.0: S_k2 = 0.0
        if R_k2 < 0.0: R_k2 = 0.0
        
        dS_k2 = g_s_eff * S_k2 * (K_CAPACITY - (S_k2 + c * R_k2)) / K_CAPACITY - a_eff * S_k2 + b * R_k2 - d_th * S_k2
        dR_k2 = g_r     * R_k2 * (K_CAPACITY - (d * S_k2 + R_k2)) / K_CAPACITY + a_eff * S_k2 - b * R_k2
        
        # --- k3 ---
        S_k3 = S_curr + 0.5 * dt * dS_k2
        R_k3 = R_curr + 0.5 * dt * dR_k2
        if S_k3 < 0.0: S_k3 = 0.0
        if R_k3 < 0.0: R_k3 = 0.0
        
        dS_k3 = g_s_eff * S_k3 * (K_CAPACITY - (S_k3 + c * R_k3)) / K_CAPACITY - a_eff * S_k3 + b * R_k3 - d_th * S_k3
        dR_k3 = g_r     * R_k3 * (K_CAPACITY - (d * S_k3 + R_k3)) / K_CAPACITY + a_eff * S_k3 - b * R_k3
        
        # --- k4 ---
        S_k4 = S_curr + dt * dS_k3
        R_k4 = R_curr + dt * dR_k3
        if S_k4 < 0.0: S_k4 = 0.0
        if R_k4 < 0.0: R_k4 = 0.0
        
        dS_k4 = g_s_eff * S_k4 * (K_CAPACITY - (S_k4 + c * R_k4)) / K_CAPACITY - a_eff * S_k4 + b * R_k4 - d_th * S_k4
        dR_k4 = g_r     * R_k4 * (K_CAPACITY - (d * S_k4 + R_k4)) / K_CAPACITY + a_eff * S_k4 - b * R_k4
        
        # Final update
        S = S_curr + (dt / 6.0) * (dS_k1 + 2.0 * dS_k2 + 2.0 * dS_k3 + dS_k4)
        R = R_curr + (dt / 6.0) * (dR_k1 + 2.0 * dR_k2 + 2.0 * dR_k3 + dR_k4)
        
        if S < 0.0: S = 0.0
        if R < 0.0: R = 0.0

    # Save final steady state populations
    out_states[idx, 0] = S
    out_states[idx, 1] = R


# --- 3. PARAMETER GENERATION (MATCHING R CODE EXACTLY + DIP DELTA) ---
def generate_parameters():
    print("  Generating parameters...")
    t0 = time.time()
    
    # Latin hypercube sampling-linear space
    x = np.array([[0.5**i, 1] for i in np.arange(1, 4)])
    y = np.array([[1, 0.5**i] for i in np.arange(0, 4)])
    wg = np.append(x, y, axis=0).round(decimals=3)

    growth_rate_fast = (1/20) * wg
    growth_rate_slow = (1/50) * wg
    comp_strong = np.asarray([[2, 2*i] for i in np.arange(0.5, 2.1, 0.1)]).round(decimals=2)
    comp_weak   = np.asarray([[0.5, 0.5*i] for i in np.arange(0.5, 2.1, 0.1)]).round(decimals=2)

    wt = wg.copy()
    transition_rate_fast = (1/200) * wt
    transition_rate_slow = (1/1000) * wt

    l_bound = [growth_rate_slow.min(), growth_rate_slow.min(),
               transition_rate_slow.min(), transition_rate_slow.min(),
               comp_weak.min(), comp_weak.min(),
               1e-4]   # Lower bound for delta
               
    u_bound = [growth_rate_fast.max(), growth_rate_fast.max(),
               transition_rate_fast.max(), transition_rate_fast.max(),
               comp_strong.max(), comp_strong.max(),
               0.05]       # Upper bound for delta
               
    sample = stats.qmc.LatinHypercube(d=7).random(n=5000)
    parms_latin = stats.qmc.scale(sample, l_bound, u_bound)

    # Log uniform sampling
    r_log     = np.exp(np.random.uniform(-6, -3, 10000)).reshape(5000, 2)
    tr_log    = np.exp(np.random.uniform(-8.5, -6.5, 10000)).reshape(5000, 2)
    comp_log  = np.exp(np.random.uniform(-2, 2, 10000)).reshape(5000, 2)
    delta_log = np.exp(np.random.uniform(np.log(1e-4), np.log(0.05), 5000)).reshape(5000, 1)
    
    parms_log = np.hstack((r_log, tr_log, comp_log, delta_log))
    parms_array = np.vstack((parms_latin, parms_log))

    # Combinations of competition and transitions
    ac_tr = parms_array.copy()
    
    ac_no_tr = parms_array.copy()
    ac_no_tr[:, 2:4] = [0., 0.]  # Zero basal transitions
    
    sc_tr = parms_array.copy()
    sc_tr[:, 5] = sc_tr[:, 4]    # Symmetric competition (c=d)
    
    sc_no_tr = sc_tr.copy()
    sc_no_tr[:, 2:4] = [0., 0.]  # Zero basal transitions

    therapy_params = np.vstack((ac_tr, ac_no_tr, sc_tr, sc_no_tr))
    
    # Add delay (always 0 for these runs)
    zero_delay = np.column_stack((therapy_params, np.zeros(therapy_params.shape[0])))
    base_params = zero_delay.copy()
    
    # Define Therapy conditions: adds [g_scale, d_th]
    N = base_params.shape[0]
    
    # 1. No Therapy
    no_treat = np.column_stack((base_params, np.ones(N) * 1.0, np.zeros(N)))
    # 2. Cytostatic 1
    st1 = np.column_stack((base_params, np.ones(N) * 3.0, np.zeros(N)))
    # 3. Cytostatic 2
    st2 = np.column_stack((base_params, np.ones(N) * 10.0, np.zeros(N)))
    # 4. Cytotoxic 1
    tx1 = np.column_stack((base_params, np.ones(N) * 1.0, np.ones(N) * 0.01))
    # 5. Cytotoxic 2
    tx2 = np.column_stack((base_params, np.ones(N) * 1.0, np.ones(N) * 0.05))

    print(f"  Total simulations: {N * 5:,}")
    print(f"  Parameter generation: {time.time() - t0:.1f}s")
    
    return {
        'no-therapy-outcomes-r3.csv': no_treat,
        'CDT-cytostatic-0.33-zero-delay-r3.csv': st1,
        'CDT-cytostatic-0.1-zero-delay-r3.csv': st2,
        'CDT-cytotoxic-0.01-zero-delay-r3.csv': tx1,
        'CDT-cytotoxic-0.05-zero-delay-r3.csv': tx2
    }


# --- 4. EXECUTION ---
def run_batch_gpu(params, name):
    t_start = time.time()
    n_sims = params.shape[0]
    
    out_states = np.zeros((n_sims, 2), dtype=np.float32)

    threadsperblock = 256
    blockspergrid = math.ceil(n_sims / threadsperblock)

    # Move to GPU
    d_params = cuda.to_device(np.ascontiguousarray(params.astype(np.float32)))
    d_out_states = cuda.to_device(out_states)
    
    # Execute
    simulate_constant_therapy_kernel[blockspergrid, threadsperblock](d_params, d_out_states)
    cuda.synchronize()
    
    # Copy back
    d_out_states.copy_to_host(out_states)
    
    # Calculate outcomes
    sen = out_states[:, 0]
    res = out_states[:, 1]
    pop_size = sen + res
    res_fraction = np.zeros_like(res)
    mask = pop_size > 0
    res_fraction[mask] = res[mask] / pop_size[mask]
    
    # Build dataframe
    df = pd.DataFrame(params[:, :10], columns=['r_x', 'r_y', 'a', 'b', 'c', 'd', 'delta', 't_delay', 'g_scale', 'cell_death'])
    df['sen'] = sen
    df['res'] = res
    df['ResFraction'] = res_fraction
    df['PopSizeSS'] = pop_size
    
    # Categorize exactly like adaptive therapy, but without "Cycling" since CDT doesn't cycle
    unfavourable = np.sum(res_fraction > 0.15)
    favourable = np.sum(res_fraction <= 0.15)
    
    # Save
    out_path = os.path.join(OUT_DIR, name)
    df.to_csv(out_path, index=False)
    
    t_end = time.time()
    print(f"  [{name}]")
    print(f"    Simulations : {n_sims:,}")
    print(f"    GPU time    : {t_end - t_start:.1f}s")
    print(f"    Saved       : {out_path}")
    print(f"    Outcomes    : Unfavourable={unfavourable:,}  Favourable={favourable:,}\n")


if __name__ == '__main__':
    print("=================================================================")
    print("  Constant Dose Therapy GPU Simulation — DIP Extension")
    print("  Target: NVIDIA RTX 3050 (4 GB VRAM)")
    print("=================================================================\n")
    
    if not cuda.is_available():
        print("  ERROR: No CUDA-capable GPU detected.")
        exit(1)
        
    device = cuda.get_current_device()
    mem = cuda.current_context().get_memory_info()
    print(f"  GPU : {device.name.decode('utf-8')}")
    print(f"  VRAM: {mem[1]/1e9:.1f} GB total, {mem[0]/1e9:.1f} GB free\n")
    
    batches = generate_parameters()
    
    print("\n  Compiling CUDA kernel (first-time warmup)...")
    t0 = time.time()
    # Dummy run to compile
    d_p = cuda.to_device(np.zeros((1, 10), dtype=np.float32))
    d_o = cuda.to_device(np.zeros((1, 2), dtype=np.float32))
    simulate_constant_therapy_kernel[1, 1](d_p, d_o)
    cuda.synchronize()
    print(f"  Kernel compiled in {time.time() - t0:.1f}s\n")
    
    print("  Starting simulations...")
    print("-" * 65 + "\n")
    
    t_total_start = time.time()
    for name, params in batches.items():
        run_batch_gpu(params, name)
        
    print("=================================================================")
    print(f"  DONE. Total time: {time.time() - t_total_start:.1f}s")
    print(f"  Output: {OUT_DIR}")
    print("=================================================================")
