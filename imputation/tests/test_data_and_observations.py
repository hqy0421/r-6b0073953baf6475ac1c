"""Tests for parsing and observation construction."""

import numpy as np
import pandas as pd
import pytest

from joint_impute.data import Problem, parse_cell, build_design
from joint_impute.model import build_observations


def test_parse_cell_variants():
    assert parse_cell("12.5") == ("exact", 12.5)
    assert parse_cell(" [1, 2] ") == ("interval", 1.0, 2.0)
    assert parse_cell("(2,1)") == ("interval", 1.0, 2.0)
    assert parse_cell("") == ("missing",)
    assert parse_cell("NA") == ("missing",)
    assert parse_cell(np.nan) == ("missing",)
    with pytest.raises(ValueError):
        parse_cell("not-a-number")


def _toy_problem():
    df = pd.DataFrame({
        "id": ["a", "b", "c"],
        "region": ["X", "X", "Y"],
        "industry": ["m", "t", "m"],
        "employees": [100.0, 200.0, 150.0],
        # Row a exact R=100/C=80 (profit 20); row b revenue interval [40,60];
        # row c revenue missing, cost interval [20,30].
        "revenue": ["100", "[40,60]", ""],
        "cost": ["80", "", "[20,30]"],
        # Row a's profit is implied by its two exact cells (deduped as a
        # redundant equality); row c's published profit is an independent
        # constraint because its revenue is missing.
        "profit": [20.0, np.nan, 30.0],
    })
    df["revenue_spec"] = df["revenue"].map(parse_cell)
    df["cost_spec"] = df["cost"].map(parse_cell)
    X, names = build_design(df, numeric_features=["employees"],
                            categorical_features=["region", "industry"])
    # Totals consistent with the exact + interval cells:
    # region X revenue total covers 100 plus row b's interval -> 150;
    # region Y contains only the missing row -> 50.
    totals = pd.DataFrame({
        "region": ["X", "Y"],
        "total_revenue": [150.0, 50.0],
    })
    prob = Problem(df=df, X=X, feature_names=names, totals=totals)
    prob.totals_group_cols = ["region"]
    return prob


def test_observations_include_profit_and_totals():
    prob = _toy_problem()
    obs = build_observations(prob)
    labels = " ".join(obs.eq_labels)
    assert "exact revenue id=a" in labels
    assert "exact cost id=a" in labels
    assert "published profit id=c" in labels
    assert "total revenue group=('X',)" in labels
    assert "total revenue group=('Y',)" in labels
    assert len(obs.interval_rows) == 2
    # Profit row: R_c - C_c = 30
    profit_row = obs.A_eq[obs.eq_labels.index("published profit id=c")]
    c_idx = 2 * 2
    assert profit_row[c_idx] == 1 and profit_row[c_idx + 1] == -1
    assert obs.b_eq[obs.eq_labels.index("published profit id=c")] == 30.0


def test_inconsistent_published_values_raise():
    prob = _toy_problem()
    # Row a has exact R=100 and exact C=80, so its profit is fixed at 20.
    # Publishing profit 999 contradicts the two exact cells directly.
    prob.df.loc[0, "profit"] = 999.0
    with pytest.raises(ValueError, match="inconsistent"):
        build_observations(prob)


def test_inconsistent_total_raises():
    prob = _toy_problem()
    # Region Y contains a single row whose exact revenue is not published,
    # so instead contradict the exact revenue cell directly: make region X's
    # total equal to row a's exact revenue while demanding a different value
    # for that same cell through a second, conflicting published datum.
    # Simpler robust case: duplicate region X with an incompatible total.
    extra = prob.totals[prob.totals["region"] == "X"].copy()
    extra["total_revenue"] = prob.totals.loc[extra.index, "total_revenue"] + 1.0
    prob.totals = pd.concat([prob.totals, extra], ignore_index=True)
    with pytest.raises(ValueError, match="inconsistent"):
        build_observations(prob)


def test_redundant_consistent_constraints_deduped():
    prob = _toy_problem()
    # Same total appended twice: consistent duplicate must not error.
    prob.totals = pd.concat([prob.totals, prob.totals.iloc[[0]]], ignore_index=True)
    obs = build_observations(prob)
    assert np.linalg.matrix_rank(obs.A_eq) == obs.A_eq.shape[0]
