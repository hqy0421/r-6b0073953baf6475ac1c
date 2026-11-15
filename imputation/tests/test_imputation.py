"""End-to-end tests: constraints hold for every completed dataset."""

import numpy as np
import pandas as pd

from joint_impute.data import Problem, build_design, parse_cell
from joint_impute.model import build_observations, equality_nullspace, mle_fit
from joint_impute.impute import (
    run_multiple_imputation,
    verify_draws,
    _linear_bounds,
    _feasible_init,
)
from scripts.make_demo import generate


def _demo_problem(seed=11, n=20):
    panel, totals = generate(seed=seed, n=n)
    panel["revenue_spec"] = panel["revenue"].map(parse_cell)
    panel["cost_spec"] = panel["cost"].map(parse_cell)
    X, names = build_design(panel, numeric_features=["employees", "assets"],
                            categorical_features=["industry", "region", "year"])
    prob = Problem(df=panel, X=X, feature_names=names, totals=totals)
    prob.totals_group_cols = ["region"]
    return prob, panel, totals


def test_every_draw_satisfies_all_identities():
    prob, panel, totals = _demo_problem()
    fit = mle_fit(prob)
    result = run_multiple_imputation(
        prob, fit, n_chains=2, n_draws=40, burn_in=300, thin=2, seed=42)

    checks = result["diagnostics"]["verification"]
    assert checks["all_constraints_satisfied"], checks
    assert checks["max_equality_residual"] < 1e-7
    assert checks["interval_violations"] == 0
    assert checks["negative_values"] == 0

    z = result["draws_z"]
    obs = fit["observations"]

    # Exact cells are literally unchanged across every draw.
    df = prob.df
    for i, spec in enumerate(df["revenue_spec"]):
        if spec[0] == "exact":
            assert np.all(np.abs(z[:, 2 * i] - spec[1]) < 1e-9)
    for i, spec in enumerate(df["cost_spec"]):
        if spec[0] == "exact":
            assert np.all(np.abs(z[:, 2 * i + 1] - spec[1]) < 1e-9)

    # Profit identity holds for every row of every draw.
    profits = z[:, 0::2] - z[:, 1::2]
    for i, p in enumerate(df["profit"]):
        if not np.isnan(p):
            assert np.all(np.abs(profits[:, i] - p) < 1e-8)

    # Regional totals hold for every draw.
    rev = z[:, 0::2]
    for _, trow in totals.iterrows():
        idx = df.index[df["region"] == trow["region"]].to_numpy()
        assert np.all(np.abs(rev[:, idx].sum(axis=1) - trow["total_revenue"]) < 1e-7)

    # Every interval contains its imputed values.
    for i, spec in enumerate(df["revenue_spec"]):
        if spec[0] == "interval":
            assert np.all(rev[:, i] >= spec[1] - 1e-9)
            assert np.all(rev[:, i] <= spec[2] + 1e-9)


def test_nullspace_and_feasible_start():
    prob, _, _ = _demo_problem(n=12)
    obs = build_observations(prob)
    x0, K = equality_nullspace(obs)
    assert np.linalg.norm(obs.A_eq @ x0 - obs.b_eq) < 1e-8
    assert np.linalg.norm(obs.A_eq @ K) < 1e-8
    rng = np.random.default_rng(0)
    u = _feasible_init(obs, x0, K, rng)
    M, lows, highs = _linear_bounds(obs, x0, K)
    v = M @ u
    assert np.all(v >= lows - 1e-8) and np.all(v <= highs + 1e-8)


def test_posterior_covers_true_values_and_intervals():
    prob, panel, _ = _demo_problem(seed=23, n=40)
    fit = mle_fit(prob)
    result = run_multiple_imputation(
        prob, fit, n_chains=3, n_draws=120, burn_in=600, thin=2, seed=7)
    summary = result["summary"].set_index("id")
    _, K = equality_nullspace(build_observations(prob))

    # Only cells the equality system leaves free have a non-degenerate
    # posterior; pinned cells are deterministic and excluded from coverage.
    free_r = {panel["id"].iloc[i] for i in range(prob.n)
              if np.linalg.norm(K[2 * i]) > 1e-10}
    free_c = {panel["id"].iloc[i] for i in range(prob.n)
              if np.linalg.norm(K[2 * i + 1]) > 1e-10}

    covered_r = covered_c = checked_r = checked_c = 0
    for _, row in panel.iterrows():
        s = summary.loc[row["id"]]
        if row["revenue_spec"][0] != "exact" and row["id"] in free_r:
            checked_r += 1
            if s["revenue_q05"] <= row["_true_revenue"] <= s["revenue_q95"]:
                covered_r += 1
            # The band from an interval disclosure must always contain truth.
            if row["revenue_spec"][0] == "interval":
                lo, hi = row["revenue_spec"][1], row["revenue_spec"][2]
                assert lo <= row["_true_revenue"] <= hi
        if row["cost_spec"][0] != "exact" and row["id"] in free_c:
            checked_c += 1
            if s["cost_q05"] <= row["_true_cost"] <= s["cost_q95"]:
                covered_c += 1

    # 90% credible intervals over a finite Monte Carlo sample: require a
    # pragmatic majority coverage rather than the nominal 0.9 point estimate.
    assert checked_r >= 5 and checked_c >= 5
    assert covered_r / checked_r > 0.6
    assert covered_c / checked_c > 0.6

    # Correlation learning: data were generated with rho = 0.85.
    assert result["rho_draws"].mean() > 0.4
    assert result["diagnostics"]["max_split_rhat"] < 1.5


def test_rubin_fraction_missing_info_nonnegative():
    prob, _, _ = _demo_problem(n=16)
    fit = mle_fit(prob)
    result = run_multiple_imputation(
        prob, fit, n_chains=2, n_draws=40, burn_in=300, thin=2, seed=99)
    for pool in result["rubin"].values():
        assert pool["total_variance"] >= pool["within_variance"] - 1e-12
        assert 0 <= pool["fraction_missing_info"] <= 1.0001
        assert pool["estimate"] > 0

    # Per-cell Rubin decomposition: at least one genuinely free cell must carry
    # positive between-imputation variance (the signal that the multiple
    # completions differ), while the pooled total variance is never below the
    # within term.  Pinned cells have FMI = 0 by construction.
    rc = result["rubin_cells"]
    assert (rc["revenue_between_variance"] > 1e-9).any()
    assert np.all(rc["revenue_total_variance"]
                  >= rc["revenue_within_variance"] - 1e-9)
    assert np.all((rc["revenue_fraction_missing_info"] >= 0)
                  & (rc["revenue_fraction_missing_info"] <= 1.0001))


def test_published_totals_pin_the_grand_mean():
    """When every row belongs to a totaled group, the grand mean is fixed.

    This is a deliberate consequence of honoring the published regional
    totals exactly: the between-imputation variance of the grand mean is then
    zero even though individual imputed cells differ across chains.
    """
    prob, _, _ = _demo_problem(seed=5, n=24)
    fit = mle_fit(prob)
    result = run_multiple_imputation(
        prob, fit, n_chains=3, n_draws=40, burn_in=400, thin=2, seed=5)
    pool = result["rubin"]["grand_mean_revenue"]
    assert pool["between_variance"] < 1e-6
    # ...but the underlying completed datasets are not identical.
    assert result["draws_z"].std(axis=0).sum() > 1e-6


def test_uncertainty_is_kept_not_collapsed():
    prob, panel, _ = _demo_problem(seed=5, n=24)
    fit = mle_fit(prob)
    result = run_multiple_imputation(
        prob, fit, n_chains=2, n_draws=50, burn_in=400, thin=2, seed=5)
    summary = result["summary"].set_index("id")
    obs = build_observations(prob)
    _, K = equality_nullspace(obs)
    # A non-exact cell can carry genuine uncertainty only where the equality
    # system leaves the coordinate free; cells pinned by profit + total + exact
    # constraints are (correctly) deterministic and have zero posterior spread.
    imputed_ids = panel.loc[
        panel["revenue_spec"].map(lambda s: s[0] != "exact"), "id"]
    free_ids = [
        panel["id"].iloc[i] for i in range(prob.n)
        if np.linalg.norm(K[2 * i]) > 1e-10
    ]
    test_ids = [i for i in imputed_ids if i in free_ids]
    assert len(test_ids) >= 3, "fixture should leave several cells genuinely free"
    sds = summary.loc[test_ids, "revenue_sd"]
    # Multiple distinct completions -> positive posterior spread, never zero.
    assert np.all(sds > 1e-6)
    assert result["draws_z"].shape[0] >= 100
