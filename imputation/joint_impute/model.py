"""Bivariate Gaussian regression and its joint (composite) likelihood.

Latent model for enterprise *i* with feature vector ``x_i``::

    (R_i, C_i)' ~ N(B' x_i, Sigma),   Sigma = [[s_R^2, rho s_R s_C],
                                                [rho s_R s_C, s_C^2]]

Every published datum is a linear observation on the stacked latent vector
``z = [R_0, C_0, R_1, C_1, ...]``:

* exact revenue/cost cell        -> equality  e_j' z = value
* published profit               -> equality  (R_i - C_i) = profit
* regional total                 -> equality  sum over the group = total
* interval cell [lo, hi]         -> censored linear observation

The equality observations define one jointly Gaussian likelihood; the
censored observations enter through their conditional marginal Gaussian
probabilities (a composite likelihood -- consistent for the parameters, and
all constraints are later enforced exactly at the imputation stage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal

from .data import Problem


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------

@dataclass
class Observations:
    A_eq: np.ndarray                     # (m_eq, 2n) equality coefficients
    b_eq: np.ndarray                     # (m_eq,) equality values
    eq_labels: list[str]                 # human-readable label per row
    interval_rows: list[np.ndarray] = field(default_factory=list)
    interval_bounds: list[tuple[float, float]] = field(default_factory=list)
    interval_labels: list[str] = field(default_factory=list)


def build_observations(problem: Problem) -> Observations:
    """Translate raw panel cells + totals into linear observations."""
    n = problem.n
    A_rows: list[np.ndarray] = []
    b_vals: list[float] = []
    labels: list[str] = []
    interval_rows: list[np.ndarray] = []
    interval_bounds: list[tuple[float, float]] = []
    interval_labels: list[str] = []

    df = problem.df
    for i, row in df.iterrows():
        rid = row["id"]
        r_spec, c_spec = row["revenue_spec"], row["cost_spec"]
        e_r = np.zeros(2 * n)
        e_r[2 * i] = 1.0
        e_c = np.zeros(2 * n)
        e_c[2 * i + 1] = 1.0

        if r_spec[0] == "exact":
            A_rows.append(e_r)
            b_vals.append(r_spec[1])
            labels.append(f"exact revenue id={rid}")
        elif r_spec[0] == "interval":
            interval_rows.append(e_r)
            interval_bounds.append((r_spec[1], r_spec[2]))
            interval_labels.append(f"interval revenue id={rid}")

        if c_spec[0] == "exact":
            A_rows.append(e_c)
            b_vals.append(c_spec[1])
            labels.append(f"exact cost id={rid}")
        elif c_spec[0] == "interval":
            interval_rows.append(e_c)
            interval_bounds.append((c_spec[1], c_spec[2]))
            interval_labels.append(f"interval cost id={rid}")

        # Published profit is an observation of the difference.
        if "profit" in df.columns and not np.isnan(row["profit"]):
            a = np.zeros(2 * n)
            a[2 * i] = 1.0
            a[2 * i + 1] = -1.0
            A_rows.append(a)
            b_vals.append(float(row["profit"]))
            labels.append(f"published profit id={rid}")

    obs = Observations(
        A_eq=np.array(A_rows, dtype=float).reshape(-1, 2 * n),
        b_eq=np.array(b_vals, dtype=float),
        eq_labels=labels,
        interval_rows=interval_rows,
        interval_bounds=interval_bounds,
        interval_labels=interval_labels,
    )

    if problem.totals is not None:
        _attach_totals(problem, obs)

    _check_equality_rank(obs)
    return obs


def _attach_totals(problem: Problem, obs: Observations) -> None:
    n = problem.n
    group_cols = problem.totals_group_cols
    totals = problem.totals
    panel_groups = problem.df.groupby(group_cols, dropna=False).indices

    def _lookup(key):
        if key in panel_groups:
            return panel_groups[key]
        # Single grouping column: pandas keys by the scalar rather than a tuple.
        if len(key) == 1 and key[0] in panel_groups:
            return panel_groups[key[0]]
        return None

    for _, trow in totals.iterrows():
        key = tuple(trow[c] for c in group_cols)
        idx = _lookup(key)
        if idx is None or len(idx) == 0:
            raise ValueError(f"total published for group {key} but panel has no such rows")
        a = np.zeros(2 * n)
        for i in idx:
            a[2 * i] = 1.0
        obs.A_eq = np.vstack([obs.A_eq, a])
        obs.b_eq = np.append(obs.b_eq, float(trow["total_revenue"]))
        obs.eq_labels.append(f"total revenue group={key}")
        if "total_cost" in totals.columns and not np.isnan(trow.get("total_cost", np.nan)):
            a = np.zeros(2 * n)
            for i in idx:
                a[2 * i + 1] = 1.0
            obs.A_eq = np.vstack([obs.A_eq, a])
            obs.b_eq = np.append(obs.b_eq, float(trow["total_cost"]))
            obs.eq_labels.append(f"total cost group={key}")


def _check_equality_rank(obs: Observations) -> None:
    A, b = obs.A_eq, obs.b_eq
    if A.shape[0] == 0:
        return
    # An underdetermined system (more latent variables than equalities) is the
    # normal case here, so augmented-rank comparison cannot detect conflicts:
    # use the minimum achievable least-squares residual instead.  This must run
    # against the full (not yet de-duplicated) constraint set, otherwise a
    # conflicting duplicate would be silently dropped before the check.
    x0, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ x0 - b
    # Relative residual normalized by the scale of the published values and
    # the constraint row norms.  Avoid np.linalg.cond, which is infinite for
    # rank-deficient A and would otherwise inflate the tolerance.  The 1e-4
    # threshold accommodates routine disclosure rounding (totals published at
    # fewer decimals than the line items) while still rejecting genuine
    # contradictions, whose relative residual is order one.
    row_scales = np.linalg.norm(A, axis=1) * (np.max(np.abs(b)) if np.max(np.abs(b)) > 0 else 1.0)
    rel = np.max(np.abs(resid) / np.maximum(row_scales, 1e-12))
    if rel > 1e-4:
        worst = int(np.argmax(np.abs(resid)))
        raise ValueError(
            "published observations are mutually inconsistent: constraint "
            f"'{obs.eq_labels[worst]}' cannot hold together with the others "
            f"(residual={resid[worst]:.4g})"
        )
    # Consistent system: drop linearly dependent (redundant) rows so that the
    # equality covariance below stays positive definite.
    if np.linalg.matrix_rank(A) < A.shape[0]:
        keep = _independent_rows(A)
        obs.A_eq = A[keep]
        obs.b_eq = b[keep]
        obs.eq_labels = [obs.eq_labels[i] for i in keep]


def _independent_rows(A: np.ndarray) -> list[int]:
    keep: list[int] = []
    basis = np.empty((0, A.shape[1]))
    for i in range(A.shape[0]):
        trial = np.vstack([basis, A[i]])
        if np.linalg.matrix_rank(trial) > basis.shape[0]:
            basis = trial
            keep.append(i)
    return keep


# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------

def unpack_theta(theta: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return regression matrix ``B`` (p x 2) and covariance ``Sigma`` (2 x 2)."""
    beta_r = theta[0:p]
    beta_c = theta[p:2 * p]
    log_sr, log_sc, atanh_rho = theta[2 * p:2 * p + 3]
    # Keep the scales finite and the correlation strictly inside (-1, 1) so
    # that the 2x2 covariance stays positive definite along the optimizer's
    # numerical path (the LKJ prior keeps the posterior away from the edges).
    log_sr = float(np.clip(log_sr, -30.0, 30.0))
    log_sc = float(np.clip(log_sc, -30.0, 30.0))
    # |rho| <= tanh(3.2) ~= 0.997: a hard guard against a singular Sigma,
    # which would otherwise collapse the MCMC proposals.
    atanh_rho = float(np.clip(atanh_rho, -3.2, 3.2))
    sr, sc = np.exp(log_sr), np.exp(log_sc)
    rho = np.tanh(atanh_rho)
    B = np.column_stack([beta_r, beta_c])
    Sigma = np.array([[sr**2, rho * sr * sc], [rho * sr * sc, sc**2]])
    return B, Sigma


def _initial_theta(problem: Problem) -> np.ndarray:
    """OLS-on-observed-means starting values."""
    n, p = problem.n, problem.p
    r_vals = np.array(
        [s[1] if s[0] == "exact" else (s[1] + s[2]) / 2 if s[0] == "interval" else np.nan
         for s in problem.df["revenue_spec"]]
    )
    c_vals = np.array(
        [s[1] if s[0] == "exact" else (s[1] + s[2]) / 2 if s[0] == "interval" else np.nan
         for s in problem.df["cost_spec"]]
    )

    def ols(y):
        mask = ~np.isnan(y)
        if mask.sum() >= p:
            beta, *_ = np.linalg.lstsq(problem.X[mask], y[mask], rcond=None)
            resid = problem.X[mask] @ beta - y[mask]
            scale = max(resid.std(ddof=max(1, p)), 1e-3)
        else:
            beta, scale = np.zeros(p), max(np.nanstd(y), 1e-3)
        return beta, scale

    br, sr = ols(r_vals)
    bc, sc = ols(c_vals)
    return np.concatenate([br, bc, [np.log(sr), np.log(sc), 0.0]]), np.array([sr, sc])


def ols_scale_hint(problem: Problem) -> np.ndarray:
    """Robust OLS residual scale estimates used to anchor the scale priors."""
    _, hints = _initial_theta(problem)
    return hints


# ---------------------------------------------------------------------------
# Gaussian bookkeeping
# ---------------------------------------------------------------------------

def _safe_inv2(S: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric 2x2 matrix with a tiny ridge for stability."""
    eps = 1e-10 * np.trace(S)
    return np.linalg.inv(S + eps * np.eye(2))


def latent_moments(problem: Problem, B: np.ndarray, Sigma: np.ndarray):
    """Stacked mean ``mu`` and block-diagonal precision / covariance of z."""
    n = problem.n
    means = problem.X @ B                       # (n, 2)
    mu = means.reshape(-1)
    P_block = _safe_inv2(Sigma)
    P = np.zeros((2 * n, 2 * n))
    for i in range(n):
        P[2 * i:2 * i + 2, 2 * i:2 * i + 2] = P_block
    V = np.zeros_like(P)
    for i in range(n):
        V[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Sigma
    return mu, V, P


def equality_partition(mu: np.ndarray, V: np.ndarray, obs: Observations):
    """Gaussian moments of the equality block and its cross-covariance."""
    A = obs.A_eq
    m_E = A @ mu
    V_EE = A @ V @ A.T
    jitter = 1e-10 * np.trace(V_EE) / max(A.shape[0], 1)
    V_EE = V_EE + jitter * np.eye(A.shape[0])
    return m_E, V_EE


def _log_interval_terms(problem, obs: Observations, mu, V, m_E, V_EE) -> float:
    if not obs.interval_rows:
        return 0.0
    L = np.array(obs.interval_rows)                  # (k, 2n)
    A = obs.A_eq
    m_L = L @ mu
    V_LL = np.einsum("ij,jk,ik->i", L, V, L)
    V_LE = L @ V @ A.T
    V_E_inv = np.linalg.inv(V_EE)
    cond_mean = m_L + V_LE @ (V_E_inv @ (obs.b_eq - m_E))
    cond_var = V_LL - np.einsum("ij,jk,ik->i", V_LE, V_E_inv, V_LE)
    cond_var = np.maximum(cond_var, 1e-12)
    sd = np.sqrt(cond_var)
    total = 0.0
    for (lo, hi), m, s in zip(obs.interval_bounds, cond_mean, sd):
        a, b = (lo - m) / s, (hi - m) / s
        total += _log_diff_ndtr(a, b)
    return total


def _log_diff_ndtr(a: float, b: float) -> float:
    """Stable log(Phi(b) - Phi(a)) for b > a, including wide tails."""
    if b < 0:
        return log_ndtr(b) + np.log1p(-np.exp(log_ndtr(a) - log_ndtr(b)))
    if a > 0:
        # Phi(b) - Phi(a) = Phi(-a) - Phi(-b)
        return log_ndtr(-a) + np.log1p(-np.exp(log_ndtr(-b) - log_ndtr(-a)))
    return np.log(max(_norm_sf(a) - _norm_sf(b), 1e-300))


def _norm_sf(x: float) -> float:
    from scipy.stats import norm
    return norm.sf(x)


# ---------------------------------------------------------------------------
# Joint MAP fit
# ---------------------------------------------------------------------------

def _log_prior(theta: np.ndarray, p: int, scale_hint: np.ndarray) -> float:
    beta = theta[:2 * p]
    log_sr, log_sc, atanh_rho = theta[2 * p:2 * p + 3]
    rho = np.tanh(np.clip(atanh_rho, -3.2, 3.2))
    ridge = -0.5 * np.sum(beta**2) / 100.0
    # Half-normal priors on the two residual scales, anchored to OLS residual
    # scale estimates so the penalty is in the data's own units.  This rules
    # out the unbounded-variance ridge that otherwise destabilizes the fit.
    sr, sc = np.exp(np.clip(log_sr, -30, 30)), np.exp(np.clip(log_sc, -30, 30))
    scale_prior = -0.5 * ((sr / scale_hint[0]) ** 2 + (sc / scale_hint[1]) ** 2)
    # LKJ(eta=4) on the 2x2 correlation matrix: p(rho) proportional to
    # (1-rho^2)^3.  Strong enough to keep rho off the +/-1 boundary (where
    # Sigma would be singular and MCMC proposals collapse) while still letting
    # strongly correlated revenue/cost data (e.g. rho=0.85) come through.
    lkj = 3.0 * np.log(max(1 - rho**2, 1e-12))
    return ridge + scale_prior + lkj


def neg_log_posterior(theta: np.ndarray, problem: Problem, obs: Observations,
                      scale_hint: np.ndarray) -> float:
    if not np.all(np.isfinite(theta)):
        return 1e25
    B, Sigma = unpack_theta(theta, problem.p)
    mu, V, _ = latent_moments(problem, B, Sigma)
    ll = 0.0
    if obs.A_eq.shape[0]:
        m_E, V_EE = equality_partition(mu, V, obs)
        try:
            ll += multivariate_normal.logpdf(obs.b_eq, mean=m_E, cov=V_EE, allow_singular=True)
        except np.linalg.LinAlgError:
            return 1e25
        ll += _log_interval_terms(problem, obs, mu, V, m_E, V_EE)
    else:
        # No equality data: intervals use unconditional marginals.
        L = np.array(obs.interval_rows) if obs.interval_rows else np.zeros((0, 2 * problem.n))
        if len(obs.interval_rows):
            m_L, sd = L @ mu, np.sqrt(np.einsum("ij,jk,ik->i", L, V, L))
            for (lo, hi), m, s in zip(obs.interval_bounds, m_L, sd):
                ll += _log_diff_ndtr((lo - m) / s, (hi - m) / s)
    value = -(ll + _log_prior(theta, problem.p, scale_hint))
    return value if np.isfinite(value) else 1e25


def mle_fit(problem: Problem, obs: Observations | None = None) -> dict:
    """Joint MAP estimate of (B, Sigma).

    Returns theta, unpacked parameters, optimizer result and the observation
    set so downstream code rebuilds nothing.
    """
    if obs is None:
        obs = build_observations(problem)
    theta0, scale_hint = _initial_theta(problem)
    best = None
    for start in (theta0, theta0 + 0.1 * np.random.default_rng(0).standard_normal(theta0.size)):
        res = minimize(
            neg_log_posterior,
            start,
            args=(problem, obs, scale_hint),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if best is None or res.fun < best.fun:
            best = res
    B, Sigma = unpack_theta(best.x, problem.p)
    return {
        "theta": best.x,
        "B": B,
        "Sigma": Sigma,
        "result": best,
        "observations": obs,
        "log_posterior": -best.fun,
    }


def constrained_gaussian_conditional(problem: Problem, fit: dict):
    """Posterior Gaussian of z given equalities at the fitted parameters.

    Used to initialize the MCMC chains.  Interval constraints are *not*
    imposed here (the sampler enforces them as hard bounds).
    """
    obs = fit["observations"]
    B, Sigma = fit["B"], fit["Sigma"]
    mu, V, _ = latent_moments(problem, B, Sigma)
    A = obs.A_eq
    if A.shape[0] == 0:
        return mu, V
    m_E, V_EE = equality_partition(mu, V, obs)
    V_inv = np.linalg.inv(V_EE)
    V_zE = V @ A.T
    cond_mean = mu + V_zE @ (V_inv @ (obs.b_eq - m_E))
    cond_cov = V - V_zE @ V_inv @ A @ V
    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    return cond_mean, cond_cov


def equality_nullspace(obs: Observations):
    """Particular solution x0 and basis K of the equality-feasible affine set."""
    A, b = obs.A_eq, obs.b_eq
    if A.shape[0] == 0:
        return b.reshape(-1) if b.size else np.zeros(A.shape[1]), np.eye(A.shape[1])
    x0, *_ = np.linalg.lstsq(A, b, rcond=None)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    tol = max(A.shape) * np.spacing(s[0])
    rank = int(np.sum(s > tol))
    # Rows of vh beyond `rank` are the right-singular vectors for (near-)zero
    # singular values, i.e. a basis of the null space of A.
    K = vh[rank:].T
    return x0, K
