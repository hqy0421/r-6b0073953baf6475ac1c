"""Constrained Bayesian multiple imputation.

The sampler alternates two Gibbs steps:

1. **Latent block** -- revenue/cost live on the affine set that satisfies
   *every* equality observation exactly (exact cells, published profits,
   regional totals): ``z = x0 + K u`` where ``K`` is the null-space basis of
   the stacked equality matrix.  Metropolis proposals for ``u`` are rejected
   whenever an interval or a non-negativity bound is violated, so each retained
   draw honours all constraints.  There is no post-hoc adjustment of rows.
2. **Parameter block** -- ``(B, Sigma)`` of the bivariate Gaussian regression
   are drawn from the Normal-Inverse-Wishart posterior given the completed
   data, so imputation uncertainty includes parameter uncertainty (including
   the learned revenue/cost correlation rho).

Several dispersed chains are run; split-R-hat diagnostics and multiple
completed datasets (Rubin's rules) are returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import invwishart

from .data import Problem
from .model import (
    Observations,
    constrained_gaussian_conditional,
    equality_nullspace,
    latent_moments,
)


# ---------------------------------------------------------------------------
# Feasibility and bounds
# ---------------------------------------------------------------------------

def _linear_bounds(obs: Observations, x0: np.ndarray, K: np.ndarray):
    """Express every interval as linear bounds ``lo <= M u <= hi``."""
    rows, lows, highs = [], [], []
    if obs.interval_rows:
        L = np.array(obs.interval_rows)
        M = L @ K
        offset = L @ x0
        for (lo, hi), m, c in zip(obs.interval_bounds, M, offset):
            rows.append(m)
            lows.append(lo - c)
            highs.append(hi - c)
    # Domain knowledge: revenue and cost are non-negative.
    n2 = K.shape[0]
    for j in range(n2):
        e = np.zeros(n2)
        e[j] = 1.0
        rows.append(e @ K)
        lows.append(-e @ x0)
        highs.append(np.inf)
    return np.array(rows), np.array(lows), np.array(highs)


def _feasible_init(obs: Observations, x0: np.ndarray, K: np.ndarray, rng) -> np.ndarray:
    """Find a starting u satisfying all interval / positivity bounds."""
    M, lows, highs = _linear_bounds(obs, x0, K)
    dim = K.shape[1]

    def objective(u):
        return 0.5 * np.sum(u**2)

    def jacobian(u):
        return u

    constraints = []
    for row, lo, hi in zip(M, lows, highs):
        if np.isfinite(lo):
            constraints.append({"type": "ineq", "fun": (lambda u, r=row, b=lo: r @ u - b),
                                "jac": (lambda u, r=row: r)})
        if np.isfinite(hi):
            constraints.append({"type": "ineq", "fun": (lambda u, r=row, b=hi: b - r @ u),
                                "jac": (lambda u, r=row: -r)})
    res = minimize(objective, np.zeros(dim), jac=jacobian,
                   constraints=constraints, method="SLSQP",
                   options={"maxiter": 1000, "ftol": 1e-10})
    # SLSQP sometimes returns a feasible point while flagging a linesearch
    # warning, so judge by the actual constraint violation rather than its
    # success flag.  A second polish run from that point usually converges.
    if _bounds_hold(res.x, M, lows, highs):
        return res.x
    polish = minimize(objective, res.x, jac=jacobian,
                      constraints=constraints, method="SLSQP",
                      options={"maxiter": 1000, "ftol": 1e-12})
    if _bounds_hold(polish.x, M, lows, highs):
        return polish.x
    # Last resort: random search from the origin.
    for _ in range(2000):
        u = rng.normal(scale=0.1, size=dim)
        if _bounds_hold(u, M, lows, highs):
            return u
    raise RuntimeError("could not find a constraint-feasible starting point; "
                       "check that published intervals and totals are compatible")


def _bounds_hold(u: np.ndarray, M: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> bool:
    v = M @ u
    return bool(np.all(v >= lows - 1e-8) and np.all(v <= highs + 1e-8))


def _dispersed_feasible_init(M, lows, highs, dim, rng, base=None,
                             gauss_mean=None, gauss_chol=None):
    """A random feasible starting point spread across the constrained region.

    Preferred path: draw from the equality-constrained posterior Gaussian (when
    its moments are supplied) and reject draws that violate the interval /
    positivity bounds -- these starts sit in the high-probability region and
    are naturally well separated.  Falls back to a boundary walk otherwise.
    """
    if gauss_mean is not None and gauss_chol is not None:
        for _ in range(2000):
            u = gauss_mean + gauss_chol @ rng.standard_normal(dim)
            if _bounds_hold(u, M, lows, highs):
                return u
    if base is None:
        base = np.zeros(dim)
    direction = rng.standard_normal(dim)
    direction /= max(np.linalg.norm(direction), 1e-12)
    proj = M @ direction
    base_v = M @ base
    # Largest t >= 0 keeping lows <= base_v + t*proj <= highs.
    t_hi = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = proj > 1e-15
        if np.any(pos):
            t_hi = min(t_hi, np.min((highs[pos] - base_v[pos]) / proj[pos]))
        neg = proj < -1e-15
        if np.any(neg):
            t_hi = min(t_hi, np.min((lows[neg] - base_v[neg]) / proj[neg]))
    if not np.isfinite(t_hi) or t_hi <= 0:
        return base
    return base + rng.uniform(0.3, 0.9) * t_hi * direction


def _is_feasible(u: np.ndarray, M: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> bool:
    v = M @ u
    return bool(np.all(v >= lows - 1e-8) and np.all(v <= highs + 1e-8))


# ---------------------------------------------------------------------------
# Normal-Inverse-Wishart parameter updates
# ---------------------------------------------------------------------------

@dataclass
class NIWPrior:
    B0: np.ndarray          # (p, 2) prior regression mean
    V0_inv: np.ndarray      # (p, p) prior row precision
    nu0: float
    Psi0: np.ndarray        # (2, 2) prior scale matrix


def make_weak_prior(problem: Problem, fit: dict) -> NIWPrior:
    p = problem.p
    Sigma = fit["Sigma"]
    return NIWPrior(
        B0=np.zeros((p, 2)),
        V0_inv=np.eye(p) / 100.0,
        nu0=5.0,
        Psi0=5.0 * Sigma,
    )


def _safe_cholesky(M: np.ndarray) -> np.ndarray:
    """Cholesky of a symmetric PSD matrix with an adaptive ridge if needed."""
    M = 0.5 * (M + M.T)
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        ridge = 1e-10 * max(np.trace(M) / M.shape[0], 1.0)
        for _ in range(6):
            try:
                return np.linalg.cholesky(M + ridge * np.eye(M.shape[0]))
            except np.linalg.LinAlgError:
                ridge *= 10.0
        # Fall back to an eigen-decomposition based PSD square root.
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, 0.0)
        return V * np.sqrt(w) @ V.T


def sample_parameters(problem: Problem, Z: np.ndarray, prior: NIWPrior, rng):
    """One N-I-W draw of (B, Sigma) given the completed n x 2 data Z."""
    X = problem.X
    XtX = X.T @ X
    XtZ = X.T @ Z
    Vn = np.linalg.inv(XtX + prior.V0_inv)
    Bn = Vn @ (XtZ + prior.V0_inv @ prior.B0)
    resid = Z - X @ Bn
    diff = Bn - prior.B0
    Psi_n = prior.Psi0 + resid.T @ resid + diff.T @ prior.V0_inv @ diff
    nu_n = prior.nu0 + X.shape[0]
    Sigma = invwishart.rvs(df=nu_n, scale=0.5 * (Psi_n + Psi_n.T), random_state=rng)
    # B | Sigma, Z ~ MatrixNormal(Bn, Vn, Sigma)
    Lv = _safe_cholesky(Vn)
    Ls = _safe_cholesky(Sigma)
    E = rng.standard_normal((X.shape[1], 2))
    B = Bn + Lv @ E @ Ls.T
    return B, 0.5 * (Sigma + Sigma.T)


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

def _safe_inverse(M: np.ndarray) -> np.ndarray:
    M = 0.5 * (M + M.T)
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        ridge = 1e-10 * max(np.trace(M) / M.shape[0], 1.0)
        return np.linalg.inv(M + ridge * np.eye(M.shape[0]))


def _u_prior_moments(mu: np.ndarray, V: np.ndarray, x0: np.ndarray, K: np.ndarray):
    """Gaussian moments of u under z ~ N(mu, V) on z = x0 + K u."""
    mean_u = K.T @ (mu - x0)
    cov_u = K.T @ V @ K
    cov_u = 0.5 * (cov_u + cov_u.T)
    return mean_u, cov_u


def run_chain(problem: Problem, obs: Observations, fit: dict, prior: NIWPrior,
              n_iter: int, burn_in: int, thin: int, chain_id: int, rng) -> dict:
    x0, K = equality_nullspace(obs)
    M, lows, highs = _linear_bounds(obs, x0, K)

    # Equality-constrained posterior Gaussian at the MAP, expressed in the
    # null-space coordinates u (z = x0 + K u).  Used both for dispersed,
    # feasible chain starts and for the Metropolis proposal.
    cond_mean, cond_cov = constrained_gaussian_conditional(problem, fit)
    mean_u0 = K.T @ (cond_mean - x0)
    prop_cov = K.T @ cond_cov @ K + 1e-10 * np.eye(K.shape[1])
    prop_L = _safe_cholesky(prop_cov)

    base = _feasible_init(obs, x0, K, rng)
    # Dispersed starts: each chain explores from a different feasible point so
    # the between-chain (between-imputation) variance is measured honestly.
    u = _dispersed_feasible_init(M, lows, highs, K.shape[1], rng, base=base,
                                 gauss_mean=mean_u0, gauss_chol=prop_L)

    B, Sigma = fit["B"], fit["Sigma"]
    mu, V, _ = latent_moments(problem, B, Sigma)
    mean_u, cov_u = _u_prior_moments(mu, V, x0, K)
    prec_u = _safe_inverse(cov_u)

    step = 0.5
    draws_z: list[np.ndarray] = []
    draws_theta: list[tuple] = []
    accepts, window = 0, 0
    target_rate = 0.234
    all_scalars: list[np.ndarray] = []

    for it in range(n_iter):
        # --- latent block: adapted random walk in the feasible space ---
        u_star = u + step * prop_L @ rng.standard_normal(K.shape[1])
        if _is_feasible(u_star, M, lows, highs):
            d = u_star - mean_u
            d0 = u - mean_u
            log_alpha = -0.5 * (d @ prec_u @ d - d0 @ prec_u @ d0)
            if np.log(rng.random()) < log_alpha:
                u = u_star
                accepts += 1
        window += 1

        # --- parameter block: conjugate N-I-W update ---
        z = x0 + K @ u
        Z = z.reshape(problem.n, 2)
        B, Sigma = sample_parameters(problem, Z, prior, rng)
        mu, V, _ = latent_moments(problem, B, Sigma)
        mean_u, cov_u = _u_prior_moments(mu, V, x0, K)
        prec_u = _safe_inverse(cov_u)

        # Robbins-Monro style adaptation during burn-in only.
        if it < burn_in and window == 50:
            rate = accepts / window
            step *= np.exp(min(0.5, max(-0.5, (rate - target_rate))))
            step = float(np.clip(step, 1e-4, 10.0))
            accepts, window = 0, 0

        if it >= burn_in and (it - burn_in) % thin == 0:
            draws_z.append(z.copy())
            draws_theta.append((B.copy(), Sigma.copy()))
        all_scalars.append(np.array([z[0::2].mean(), z[1::2].mean(), z[0], z[1]]))

    rate = accepts / max(window, 1)
    return {
        "chain_id": chain_id,
        "z": np.array(draws_z),
        "theta": draws_theta,
        "acceptance": rate,
        "step": step,
        "trace": np.array(all_scalars),
        "x0": x0,
        "K": K,
    }


def split_rhat(traces: list[np.ndarray]) -> np.ndarray:
    """Split-R-hat per scalar column from one trace per chain."""
    halves = []
    for tr in traces:
        half = len(tr) // 2
        halves.extend([tr[:half], tr[half:2 * half]])
    chains = np.stack(halves, axis=0)              # (2C, T, d)
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    overall = chain_means.mean(axis=0)
    m, t = chains.shape[:2]
    between = t / (m - 1) * np.sum((chain_means - overall) ** 2, axis=0)
    within = chain_vars.mean(axis=0)
    var_hat = (1 - 1 / t) * within + between / t
    return np.sqrt(var_hat / np.maximum(within, 1e-12))


# ---------------------------------------------------------------------------
# Verification and multiple-imputation summaries
# ---------------------------------------------------------------------------

def verify_draws(problem: Problem, obs: Observations, draws_z: np.ndarray) -> dict:
    """Assert every draw reproduces the published observations exactly."""
    A, b = obs.A_eq, obs.b_eq
    eq_err = np.abs(draws_z @ A.T - b).max() if A.shape[0] else 0.0
    interval_violations = 0
    if obs.interval_rows:
        L = np.array(obs.interval_rows)
        vals = draws_z @ L.T
        for j, (lo, hi) in enumerate(obs.interval_bounds):
            if np.any(vals[:, j] < lo - 1e-7) or np.any(vals[:, j] > hi + 1e-7):
                interval_violations += 1
    neg_count = int(np.sum(draws_z < -1e-9))
    ok = eq_err < 1e-7 and interval_violations == 0 and neg_count == 0
    return {
        "all_constraints_satisfied": ok,
        "max_equality_residual": float(eq_err),
        "interval_violations": int(interval_violations),
        "negative_values": neg_count,
        "n_draws": int(draws_z.shape[0]),
    }


def _rubin_pool(per_imputation_estimate: np.ndarray,
                per_imputation_variance: np.ndarray | None = None) -> dict:
    """Rubin's rules for a scalar estimate across ``m`` imputations.

    ``per_imputation_estimate`` has shape (m,) (or (m, k) for a vector of
    estimates).  When each imputation is itself an average over several
    posterior draws, its Monte Carlo variance (= within-imputation sampling
    variance of the mean) is supplied as ``per_imputation_variance``; this is
    the proper "within" term rather than the across-draw spread of the mean.
    """
    q = np.asarray(per_imputation_estimate, dtype=float)
    m = q.shape[0]
    qbar = q.mean(axis=0)
    between = q.var(axis=0, ddof=1)
    if per_imputation_variance is None:
        wbar = np.zeros_like(between)
    else:
        wbar = np.asarray(per_imputation_variance, dtype=float).mean(axis=0)
    total = wbar + (1 + 1 / m) * between
    r = (1 + 1 / m) * between / np.maximum(wbar, 1e-12)
    # When between-imputation variance is negligible, take r -> 0 (large df)
    # without the overflow that 1/r would produce.
    dfree = np.where(
        r > 1e-12,
        (m - 1) * (1 + 1 / np.maximum(r, 1e-12)) ** 2,
        np.inf,
    )
    return {
        "estimate": qbar,
        "within_variance": wbar,
        "between_variance": between,
        "total_variance": total,
        "fraction_missing_info": (1 + 1 / m) * between / np.maximum(total, 1e-12),
        "df": dfree,
        "n_imputations": m,
    }


def run_multiple_imputation(problem: Problem, fit: dict, *,
                            n_chains: int = 4, n_draws: int = 250,
                            burn_in: int = 1000, thin: int = 4,
                            seed: int = 20261114) -> dict:
    """Run chains, verify constraints, and build MI tables and diagnostics."""
    obs = fit["observations"]
    prior = make_weak_prior(problem, fit)
    n_iter = burn_in + n_draws * thin

    chains = []
    for cid in range(n_chains):
        rng = np.random.default_rng(seed + 1009 * cid)
        chains.append(run_chain(problem, obs, fit, prior, n_iter, burn_in,
                                thin, cid, rng))

    draws_z = np.vstack([c["z"] for c in chains])
    rhat = split_rhat([c["trace"] for c in chains])
    diagnostics = {
        "n_chains": n_chains,
        "n_draws_per_chain": n_draws,
        "burn_in": burn_in,
        "thin": thin,
        "acceptance_rates": [c["acceptance"] for c in chains],
        "final_step_sizes": [c["step"] for c in chains],
        "split_rhat_mean_revenue": float(rhat[0]),
        "split_rhat_mean_cost": float(rhat[1]),
        "split_rhat_first_row_r": float(rhat[2]),
        "split_rhat_first_row_c": float(rhat[3]),
        "max_split_rhat": float(np.max(rhat)),
        "learned_rho": float(fit["Sigma"][0, 1] /
                             np.sqrt(fit["Sigma"][0, 0] * fit["Sigma"][1, 1])),
    }
    diagnostics["verification"] = verify_draws(problem, obs, draws_z)

    # ---- long table of completed datasets ----
    n, m_total = problem.n, draws_z.shape[0]
    draws_per_chain = m_total // n_chains
    long_rows = []
    for d in range(m_total):
        rev, cost = draws_z[d, 0::2], draws_z[d, 1::2]
        chain_id = d // draws_per_chain
        for i in range(n):
            long_rows.append({
                "draw": d,
                "chain": chain_id,
                "id": problem.df["id"].iloc[i],
                "revenue": rev[i],
                "cost": cost[i],
                "profit": rev[i] - cost[i],
                "revenue_observed": problem.df["revenue_spec"].iloc[i][0] != "missing",
                "cost_observed": problem.df["cost_spec"].iloc[i][0] != "missing",
                "revenue_spec": str(problem.df["revenue_spec"].iloc[i]),
                "cost_spec": str(problem.df["cost_spec"].iloc[i]),
            })
    long_df = pd.DataFrame(long_rows)

    # ---- per-cell uncertainty summary ----
    summary_rows = []
    rev_draws = draws_z[:, 0::2]
    cost_draws = draws_z[:, 1::2]
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    for i in range(n):
        row = {"id": problem.df["id"].iloc[i],
               "revenue_spec": str(problem.df["revenue_spec"].iloc[i]),
               "cost_spec": str(problem.df["cost_spec"].iloc[i])}
        for name, arr in (("revenue", rev_draws[:, i]), ("cost", cost_draws[:, i])):
            row[f"{name}_mean"] = arr.mean()
            row[f"{name}_sd"] = arr.std(ddof=1)
            for q in qs:
                row[f"{name}_q{int(q * 100):02d}"] = np.quantile(arr, q)
        row["profit_mean"] = (rev_draws[:, i] - cost_draws[:, i]).mean()
        row["profit_sd"] = (rev_draws[:, i] - cost_draws[:, i]).std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # ---- Rubin pooling: each chain is one independent imputation set ----
    # For every chain compute the estimate (grand mean / per-cell mean) and its
    # within-chain Monte Carlo variance of the mean, then apply Rubin's rules
    # across the n_chains imputations.
    chain_z = [c["z"] for c in chains]
    chain_rev = [cz[:, 0::2] for cz in chain_z]
    chain_cost = [cz[:, 1::2] for cz in chain_z]

    def _chain_pool(chain_arrays):
        estimates = np.array([a.mean() for a in chain_arrays])
        # Variance of the chain mean = sample variance / number of draws.
        mc_var = np.array([a.var(ddof=1) / a.shape[0] for a in chain_arrays])
        return _rubin_pool(estimates, mc_var)

    pool = {
        "grand_mean_revenue": _chain_pool(chain_rev),
        "grand_mean_cost": _chain_pool(chain_cost),
    }

    # Per-cell Rubin decomposition (revenue and cost), returned as frames so
    # downstream analyses see where imputation uncertainty concentrates.
    def _cell_rubin(chain_arrays):
        stacked = np.stack(chain_arrays, axis=0)        # (m, draws, n)
        m, draws, ncells = stacked.shape
        q = stacked.mean(axis=1)                        # (m, n)
        within = stacked.var(axis=1, ddof=1) / draws    # MC var of each mean
        out = []
        for j in range(ncells):
            out.append(_rubin_pool(q[:, j], within[:, j]))
        return pd.DataFrame(out)

    rubin_r = _cell_rubin(chain_rev).add_prefix("revenue_")
    rubin_c = _cell_rubin(chain_cost).add_prefix("cost_")
    rubin_cells = pd.concat(
        [pd.DataFrame({"id": problem.df["id"].to_numpy()}), rubin_r, rubin_c], axis=1)

    # ---- parameter draws (rho etc.) for posterior reporting ----
    rho_draws = np.array([
        S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
        for ch in chains for (_, S) in ch["theta"]
    ])

    return {
        "long": long_df,
        "summary": summary_df,
        "rubin_cells": rubin_cells,
        "diagnostics": diagnostics,
        "rubin": pool,
        "rho_draws": rho_draws,
        "draws_z": draws_z,
    }
