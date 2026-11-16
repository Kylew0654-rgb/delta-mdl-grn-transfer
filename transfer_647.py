#!/usr/bin/env python3
"""
Out-of-Distribution Transfer Test for Algorithmic Common Information (Cho 647)
==============================================================================

Evaluates out-of-distribution generalization of extracted common information
(sufficient statistic alpha) in the 11-variable ERK/RKIP signaling pathway
(Cho et al., 2003; Gilbert & Heiner, 2006; BioModels BIOMD0000000647) against
topology-preserving randomized control ensembles (64 shuffled-rate and
64 uniform-rate networks).

Experimental Regimes:
  * Regime A (Training): 1000-step burn-in (u = 0.0), followed by 1000 steps
    with periodic square-wave stimulation (u = 3.5 on 200-step intervals 0, 2, 4).
  * Regime A' (Within-Regime Holdout): Continuation for 1000 steps under the
    training stimulus schedule.
  * Regime B1 (Weak/Shifted Transfer): 1000-step burn-in from initial state,
    followed by 1000 steps with attenuated, phase-shifted pulses (u = 1.2 on
    200-step intervals 1, 3).
  * Regime B2 (Random-Block Transfer): 1000-step burn-in from initial state,
    followed by 1000 steps of pseudorandom 100-step stimulus blocks
    (u in {0.0, 3.5}, fixed seed 123 across all networks).

Minimum Description Length (MDL) Transfer Metric:
  Measures compression gain (Gacs, Tromp & Vitanyi, 2001):
    Delta_MDL = sum_{i=1}^n K(X_i) - (K(alpha) + sum_{i=1}^n K(X_i | alpha))
  Compares two estimators:
    1. Baseline estimator (mdl_n_variables_train_test): fits SVD axis on X_train,
       spans quantization bounds across X_train and X_test, and selects
       partition granularity k* on X_test.
    2. Strict transfer estimator (mdl_strict_transfer): fits all parameters
       (SVD projection axis, quantization boundaries, conditional distributions,
       and optimal granularity k*) strictly on X_train and evaluates
       out-of-sample on X_test.

Usage:
  python3 transfer_647.py
"""

import time

import numpy as np

# Cho et al. (2003) / Gilbert & Heiner (2006) BIOMD0000000647 parameters & initial state
CHO_K = np.array([
    0.53, 0.0072, 0.625, 0.00245, 0.0315,
    0.8, 0.0075, 0.071, 0.92, 0.00122, 0.87,
])
CHO_Y0 = np.array([2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 2.5, 2.5, 0.0])

DT = 0.1


def rhs_rates(k_vec, y, u):
    """Computes reaction rates and state derivatives for the 11-species ODE system."""
    m = np.maximum(y, 0.0)
    r = [
        k_vec[0] * (1 + u) * m[0] * m[1],
        k_vec[1] * m[2],
        k_vec[2] * m[2] * m[8],
        k_vec[3] * m[3],
        k_vec[4] * (1 + 2 * u) * m[3],
        k_vec[5] * m[4] * m[6],
        k_vec[6] * m[7],
        k_vec[7] * (1 + 2 * u) * m[7],
        k_vec[8] * m[5] * m[9],
        k_vec[9] * m[10],
        k_vec[10] * m[10],
    ]
    return np.array([
        r[1] + r[4] - r[0],
        r[1] + r[10] - r[0],
        r[0] + r[3] - r[1] - r[2],
        r[2] - r[3] - r[4],
        r[4] + r[6] - r[5],
        r[4] + r[9] - r[8],
        r[6] + r[7] - r[5],
        r[5] - r[6] - r[7],
        r[3] + r[7] - r[2],
        r[9] + r[10] - r[8],
        r[8] - r[9] - r[10],
    ])


def run_steps(k_vec, y, n_steps, u_fn):
    """Integrates the ODE system via forward Euler for n_steps under stimulus u_fn."""
    traj = np.zeros((n_steps, 11))
    for s in range(n_steps):
        y = np.clip(y + DT * rhs_rates(k_vec, y, u_fn(s)), 0.0, 10.0)
        traj[s] = y
    return traj, y


def u_baseline(s):
    """Baseline periodic stimulus (Regimes A and A')."""
    return 3.5 if (s // 200) in (0, 2, 4) else 0.0


def u_weak_shifted(s):
    """Attenuated and phase-shifted stimulus (Regime B1)."""
    return 1.2 if (s // 200) in (1, 3) else 0.0


_RNG_B2 = np.random.default_rng(123)
U_B2_BLOCKS = _RNG_B2.choice([0.0, 3.5], size=10)


def u_random_blocks(s):
    """Random block stimulus (Regime B2)."""
    return float(U_B2_BLOCKS[s // 100])


def simulate_all(k_vec):
    """Generates trajectories for training (A), holdout (A'), and transfer (B1, B2)."""
    y = CHO_Y0.copy()
    _, y = run_steps(k_vec, y, 1000, lambda s: 0.0)
    train_A, y = run_steps(k_vec, y, 1000, u_baseline)
    test_Ap, _ = run_steps(k_vec, y, 1000, u_baseline)

    y = CHO_Y0.copy()
    _, y = run_steps(k_vec, y, 1000, lambda s: 0.0)
    test_B1, _ = run_steps(k_vec, y, 1000, u_weak_shifted)

    y = CHO_Y0.copy()
    _, y = run_steps(k_vec, y, 1000, lambda s: 0.0)
    test_B2, _ = run_steps(k_vec, y, 1000, u_random_blocks)

    return train_A, test_Ap, test_B1, test_B2


def mdl_n_variables_train_test(X_train, X_test, global_min=0.0, global_max=2.5,
                               n_bins=32, k_grid=(2, 3, 4, 6, 8, 12)):
    """Computes baseline Train -> Test MDL compression gap (Delta_MDL).

    Uses fixed concentration bounds [global_min, global_max], learns the 1D
    collective coordinate w on X_train via SVD, and evaluates across k_grid.
    """
    T_tr, n = X_train.shape
    T_te, _ = X_test.shape
    mins = np.broadcast_to(global_min, (n,)).astype(float)
    maxs = np.broadcast_to(global_max, (n,)).astype(float)
    spans = np.maximum(maxs - mins, 1e-9)

    Q_tr = np.zeros((T_tr, n), dtype=int)
    Q_te = np.zeros((T_te, n), dtype=int)
    for i in range(n):
        edges = np.linspace(mins[i] - 1e-9, maxs[i] + 1e-9, n_bins + 1)
        Q_tr[:, i] = np.clip(np.digitize(X_train[:, i], edges) - 1, 0, n_bins - 1)
        Q_te[:, i] = np.clip(np.digitize(X_test[:, i], edges) - 1, 0, n_bins - 1)

    param_unit = (0.5 * np.log2(T_tr)) / T_te
    K_Xi = np.zeros(n)
    for i in range(n):
        counts = np.bincount(Q_tr[:, i], minlength=n_bins) + 0.5
        probs = counts / np.sum(counts)
        data_bits = float(np.sum(-np.log2(probs[Q_te[:, i]]))) / T_te
        K_Xi[i] = data_bits + (n_bins - 1) * param_unit
    sum_K_Xi = float(np.sum(K_Xi))

    Z_tr = (X_train - mins) / spans
    Z_te = (X_test - mins) / spans
    mu_z = np.mean(Z_tr, axis=0)
    _, _, Vt = np.linalg.svd(Z_tr - mu_z, full_matrices=False)
    w = Vt[0]
    proj_tr = (Z_tr - mu_z) @ w
    proj_te = (Z_te - mu_z) @ w
    p_min = min(float(np.min(proj_tr)), float(np.min(proj_te))) - 1e-6
    p_max = max(float(np.max(proj_tr)), float(np.max(proj_te))) + 1e-6

    best_raw_delta = -np.inf
    best_k, best_K_alpha, best_sum_K_cond = 1, 0.0, sum_K_Xi

    for k in k_grid:
        edges_k = np.linspace(p_min, p_max, k + 1)
        a_tr = np.clip(np.digitize(proj_tr, edges_k) - 1, 0, k - 1)
        a_te = np.clip(np.digitize(proj_te, edges_k) - 1, 0, k - 1)

        a_counts = np.bincount(a_tr, minlength=k) + 0.5
        a_probs = a_counts / np.sum(a_counts)
        seq_bits = float(np.sum(-np.log2(a_probs[a_te]))) / T_te
        model_bits = (2 * k * n + n) * param_unit
        K_alpha = seq_bits + model_bits

        sum_K_cond = 0.0
        for i in range(n):
            cond_bits_i = 0.0
            for c in range(k):
                mask_tr, mask_te = (a_tr == c), (a_te == c)
                if np.any(mask_te):
                    c_counts = np.bincount(Q_tr[mask_tr, i], minlength=n_bins) + 0.5
                    c_probs = c_counts / np.sum(c_counts)
                    cond_bits_i += float(np.sum(-np.log2(c_probs[Q_te[mask_te, i]])))
            sum_K_cond += cond_bits_i / T_te

        raw_delta = sum_K_Xi - (K_alpha + sum_K_cond)
        if raw_delta > best_raw_delta:
            best_raw_delta = raw_delta
            best_k = k
            best_K_alpha = K_alpha
            best_sum_K_cond = sum_K_cond

    return {
        "sum_K_Xi": round(sum_K_Xi, 3),
        "K_alpha": round(best_K_alpha, 3),
        "sum_K_cond": round(best_sum_K_cond, 3),
        "Delta_MDL_raw": round(float(best_raw_delta), 3),
        "k_opt": int(best_k),
    }


def _quantize(X, mins, maxs, n_bins):
    """Discretizes state trajectories into uniform bins over [mins, maxs]."""
    T, n = X.shape
    Q = np.zeros((T, n), dtype=int)
    for i in range(n):
        edges = np.linspace(mins[i] - 1e-9, maxs[i] + 1e-9, n_bins + 1)
        Q[:, i] = np.clip(np.digitize(X[:, i], edges) - 1, 0, n_bins - 1)
    return Q


def _xent(counts_tr, Q_te):
    """Computes Laplace-smoothed cross-entropy code length in bits."""
    probs = (counts_tr + 0.5) / np.sum(counts_tr + 0.5)
    return float(np.sum(-np.log2(probs[Q_te])))


def mdl_strict_transfer(X_train, X_test, n_bins=32,
                        k_grid=(2, 3, 4, 6, 8, 12)):
    """Computes strict out-of-sample MDL compression gain (Delta_MDL).

    All parameters (SVD projection axis, quantization edges, conditional
    histograms, and partition size k*) are estimated strictly on X_train
    and evaluated on X_test.

    Returns:
        tuple: (delta_test, k_star, delta_train_at_kstar)
    """
    T_tr, n = X_train.shape
    T_te, _ = X_test.shape
    mins = np.zeros(n)
    maxs = np.full(n, 2.5)
    spans = maxs - mins
    Q_tr = _quantize(X_train, mins, maxs, n_bins)
    Q_te = _quantize(X_test, mins, maxs, n_bins)
    param_unit = (0.5 * np.log2(T_tr)) / T_te

    def sum_K_Xi(Q_ev, T_ev):
        tot = 0.0
        for i in range(n):
            counts = np.bincount(Q_tr[:, i], minlength=n_bins)
            tot += _xent(counts, Q_ev[:, i]) / T_ev
            tot += (n_bins - 1) * param_unit
        return tot

    Z_tr = (X_train - mins) / spans
    Z_te = (X_test - mins) / spans
    mu_z = np.mean(Z_tr, axis=0)
    _, _, Vt = np.linalg.svd(Z_tr - mu_z, full_matrices=False)
    w = Vt[0]
    proj_tr = (Z_tr - mu_z) @ w
    proj_te = (Z_te - mu_z) @ w
    p_min = float(np.min(proj_tr)) - 1e-6
    p_max = float(np.max(proj_tr)) + 1e-6

    def eval_k(k, Q_ev, proj_ev, T_ev):
        edges_k = np.linspace(p_min, p_max, k + 1)
        a_tr = np.clip(np.digitize(proj_tr, edges_k) - 1, 0, k - 1)
        a_ev = np.clip(np.digitize(proj_ev, edges_k) - 1, 0, k - 1)
        a_counts = np.bincount(a_tr, minlength=k)
        K_alpha = _xent(a_counts, a_ev) / T_ev + (2 * k * n + n) * param_unit
        sum_cond = 0.0
        for i in range(n):
            for c in range(k):
                m_tr, m_ev = (a_tr == c), (a_ev == c)
                if np.any(m_ev):
                    cc = np.bincount(Q_tr[m_tr, i], minlength=n_bins)
                    sum_cond += _xent(cc, Q_ev[m_ev, i]) / T_ev
        return K_alpha + sum_cond

    sum_tr = sum_K_Xi(Q_tr, T_tr)
    best_k, best_d_tr = 1, -np.inf
    for k in k_grid:
        d_tr = sum_tr - eval_k(k, Q_tr, proj_tr, T_tr)
        if d_tr > best_d_tr:
            best_d_tr, best_k = d_tr, k

    sum_te = sum_K_Xi(Q_te, T_te)
    d_te = sum_te - eval_k(best_k, Q_te, proj_te, T_te)
    return d_te, best_k, best_d_tr


def run_arm(name, k_vec):
    """Simulates a network parameterization and evaluates all MDL metrics."""
    tr, ap, b1, b2 = simulate_all(k_vec)
    verb = mdl_n_variables_train_test(tr, ap)['Delta_MDL_raw']
    s_ap, k_ap, _ = mdl_strict_transfer(tr, ap)
    s_b1, k_b1, _ = mdl_strict_transfer(tr, b1)
    s_b2, k_b2, _ = mdl_strict_transfer(tr, b2)
    return verb, s_ap, s_b1, s_b2


def summarize(name, rows, bio_row):
    """Prints summary statistics comparing a control ensemble against the biological network."""
    arr = np.array(rows)
    print(f"  {name}: n={len(rows)}")
    labels = [
        "baseline A->A'",
        "strict A->A'",
        "strict A->B1 (weak/shifted)",
        "strict A->B2 (random pulses)",
    ]
    for j, lab in enumerate(labels):
        col = arr[:, j]
        print(
            f"    {lab:28s} mean={col.mean():7.3f} max={col.max():7.3f} "
            f"beat bio: {int((col > bio_row[j]).sum())}/{len(col)} "
            f"(bio={bio_row[j]:.3f})"
        )


if __name__ == '__main__':
    t0 = time.time()
    bio = run_arm('bio', CHO_K)
    print(
        f"BIO: baseline A->A'={bio[0]:.3f} | "
        f"strict A->A'={bio[1]:.3f} | strict A->B1={bio[2]:.3f} | "
        f"strict A->B2={bio[3]:.3f}"
    )
    print()
    sh, un = [], []
    for s in range(64):
        rng = np.random.default_rng(1000 + s)
        sh.append(run_arm(f'shuf{s}', rng.permutation(CHO_K)))
        un.append(run_arm(f'unif{s}', rng.uniform(0.001, 1.0, 11)))
    summarize('SHUFFLED-rate twins (topology-preserving)', sh, bio)
    print()
    summarize('UNIFORM-rate twins (topology-preserving)', un, bio)
    print()
    print(f"Done in {time.time() - t0:.0f} s.")