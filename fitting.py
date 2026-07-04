import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

GMIT_N = np.array([2, 3, 4, 5, 6])
GMIT_T = np.array([0.98, 1.11, 1.46, 1.62, 1.81])
GMIT_SEM = np.array([0.2, 0.14, 0.28, 0.30, 0.27])

EPS_FIXED = 2.0
PI_FIXED = 2.5
K_FIXED = 2.5

def get_k_hindered(k_base, beta, n_inactive):
    if n_inactive <= 1: return k_base
    return k_base / (1 + beta * n_inactive * (n_inactive - 1))

def calculate_splitting_probabilities(N, k_base, beta, epsilon, pi):
    if N == 0: return np.array([1.0])
    states = [(L, j) for L in range(1, N + 1) for j in range(N - L + 1)]
    state_to_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    
    Q = np.zeros((n_states, n_states))
    R_mat = np.zeros((n_states, N))
    
    for idx, (L, j) in enumerate(states):
        n_dinact = N - L - j
        rate_unbind = L * epsilon
        
        if L == 1: R_mat[idx, j] = rate_unbind
        elif (L - 1, j) in state_to_idx: Q[idx, state_to_idx[(L - 1, j)]] += rate_unbind
            
        rate_act = 0
        if n_dinact > 0:
            rate_act = n_dinact * get_k_hindered(k_base, beta, n_dinact)
            if (L, j + 1) in state_to_idx: Q[idx, state_to_idx[(L, j + 1)]] += rate_act
                
        rate_rebind = 0
        if j > 0 and L < N:
            rate_rebind = j * pi
            if (L + 1, j - 1) in state_to_idx: Q[idx, state_to_idx[(L + 1, j - 1)]] += rate_rebind
                
        Q[idx, idx] = -(rate_unbind + rate_act + rate_rebind)
        
    try:
        Q_inv = np.linalg.inv(Q)
        return -np.dot(Q_inv, R_mat)[state_to_idx[(N, 0)], :]
    except np.linalg.LinAlgError:
        return np.zeros(N)

def calculate_MIT_vector(N, k_base, beta, pi):
    T = np.zeros(N + 1)
    if N == 0: return np.array([0.0])
    T[N] = 1.0 / (N * pi)
    
    for m in range(N - 1, -1, -1):
        n_inact = N - m
        rate_act = n_inact * get_k_hindered(k_base, beta, n_inact)
        rate_rebind = m * pi
        R_m = rate_act + rate_rebind
        T[m] = (1.0 + rate_act * T[m + 1]) / R_m if R_m > 0 else 0
    return T

def calculate_GMIT(N, epsilon, pi, k_base, beta):
    if N == 0: return 0
    P_split = calculate_splitting_probabilities(N, k_base, beta, epsilon, pi)
    T_vec = calculate_MIT_vector(N, k_base, beta, pi)
    return np.sum(P_split * T_vec[:N])

def gmit_loss(params):
    beta = params[0]
    if beta < 0: return 1e12
    chi2 = 0
    for i, N in enumerate(GMIT_N):
        chi2 += ((calculate_GMIT(int(N), EPS_FIXED, PI_FIXED, K_FIXED, beta) - GMIT_T[i]) / GMIT_SEM[i]) ** 2
    return chi2

def run_grid_scan(eps_vals, pi_vals):
    chi2_grid = np.zeros((len(eps_vals), len(pi_vals)))
    beta_grid = np.zeros_like(chi2_grid)
    best = {'chi2': np.inf}
    
    for i, eps in enumerate(eps_vals):
        for j, pi in enumerate(pi_vals):
            def local_loss(p):
                b = p[0]
                if b < 0: return 1e12
                c2 = 0
                for idx, N in enumerate(GMIT_N): c2 += ((calculate_GMIT(int(N), eps, pi, K_FIXED, b) - GMIT_T[idx]) / GMIT_SEM[idx]) ** 2
                return c2
            
            res = minimize(local_loss, [0.5], bounds=[(0, 1000)], method='L-BFGS-B')
            chi2_grid[i, j] = res.fun
            beta_grid[i, j] = res.x[0]
            
            if res.fun < best['chi2']:
                best = {'eps': eps, 'pi': pi, 'beta': res.x[0], 'chi2': res.fun}
    return best, chi2_grid, beta_grid

if __name__ == "__main__":
    res = minimize(gmit_loss, [0.5], bounds=[(0, 1000)], method='L-BFGS-B')
    beta_opt = res.x[0]
    print(f"Optimal beta: {beta_opt:.4f} (Chi2: {res.fun:.4f})")

    gmit_p = [calculate_GMIT(int(N), EPS_FIXED, PI_FIXED, K_FIXED, beta_opt) for N in GMIT_N]

    plt.figure(figsize=(8, 6))
    plt.errorbar(GMIT_N, GMIT_T, yerr=GMIT_SEM, fmt='ko', capsize=5, label='Experimental')
    plt.plot(GMIT_N, gmit_p, 'rv--', label=f'Fit (beta={beta_opt:.2f})')
    plt.title(f'GMIT Fit ($\chi^2$={res.fun:.2f})')
    plt.xlabel('N')
    plt.ylabel('GMIT (s)')
    plt.legend()
    plt.savefig('gmit_fit.png', dpi=150, bbox_inches='tight')
    plt.show()

    eps_vals = np.linspace(1.0, 4.0, 20)
    pi_vals = np.linspace(1.0, 10.0, 20)
    best_sol, chi2_map, _ = run_grid_scan(eps_vals, pi_vals)
    print(f"Grid Scan Minimum -> eps:{best_sol['eps']:.2f}, pi:{best_sol['pi']:.2f}, beta:{best_sol['beta']:.2f} (Chi2: {best_sol['chi2']:.4f})")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(chi2_map, extent=[pi_vals[0], pi_vals[-1], eps_vals[0], eps_vals[-1]], 
                   origin='lower', aspect='auto', cmap='viridis_r', vmax=np.percentile(chi2_map, 85))
    plt.colorbar(im, ax=ax, label='$\chi^2$')
    ax.plot(best_sol['pi'], best_sol['eps'], 'r*', markersize=15)
    ax.set(xlabel='Rebinding Rate $\pi$ ($s^{-1}$)', ylabel='Unbinding Rate $\epsilon$ ($s^{-1}$)', title=f'Min $\chi^2$={best_sol["chi2"]:.2f}')
    plt.show()

    max_N = 9
    results = []
    for n in range(1, max_N + 1):
        p = calculate_splitting_probabilities(n, K_FIXED, beta_opt, EPS_FIXED, PI_FIXED)
        results.append({'N': n, **{f'P(m={m})': val for m, val in enumerate(p)}})
    
    df_probs = pd.DataFrame(results).fillna(0)
    print("\nSplitting Probabilities:")
    print(df_probs.to_string(index=False))

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for i, n in enumerate(range(1, max_N + 1)):
        ax = axes.flatten()[i]
        p = df_probs.loc[df_probs['N'] == n].iloc[0, 1:].values[:n]
        ax.bar(np.arange(len(p)), p, color='skyblue', edgecolor='black')
        ax.set(title=f'N={n}', xlabel='m', ylim=(0, 1.05), xticks=np.arange(len(p)))
    plt.show()