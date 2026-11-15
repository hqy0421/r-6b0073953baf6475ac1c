"""Data loading and representation for the joint imputation pipeline.

A panel row describes one enterprise-period (for example one company-year).
Revenue/cost cells may be:

* an exact number                -> ``("exact", value)``
* an interval such as "[12,15]" -> ``("interval", lower, upper)``
* empty / "NA"                   -> ``("missing",)``

Published profits and regional totals are *additional observations*; they are
kept separate from the latent revenue/cost values and never overwrite the raw
cells.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

INTERVAL_RE = re.compile(r"^\s*[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]\s*$")
NUMBER_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-"}


def parse_cell(value) -> tuple:
    """Parse a revenue/cost cell into a specification tuple.

    Returns one of ``("exact", v)``, ``("interval", lo, hi)`` or
    ``("missing",)``.  Interval bounds are always ordered.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ("missing",)
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return ("missing",)
    m = INTERVAL_RE.match(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return ("interval", lo, hi)
    m = NUMBER_RE.match(text)
    if m:
        return ("exact", float(m.group(1)))
    raise ValueError(f"unparseable cell value: {value!r}")


def _as_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return np.nan
    return float(text)


@dataclass
class Problem:
    """Bundle of a parsed panel and its design matrix.

    Attributes
    ----------
    df:
        Panel frame with parsed ``revenue_spec`` / ``cost_spec`` columns and
        a numeric ``profit`` column (NaN when not published).
    X:
        Standardized design matrix with an intercept in column 0.
    feature_names:
        Column names of ``X``.
    totals:
        Optional frame with regional/period totals.
    """

    df: pd.DataFrame
    X: np.ndarray
    feature_names: list[str]
    totals: pd.DataFrame | None

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def latent_vector(self, revenue: np.ndarray, cost: np.ndarray) -> np.ndarray:
        """Interleaved latent layout ``[R0, C0, R1, C1, ...]``."""
        z = np.empty(2 * self.n)
        z[0::2] = revenue
        z[1::2] = cost
        return z

    def split_latent(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z[0::2].copy(), z[1::2].copy()


def build_design(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a standardized design matrix with an intercept.

    Numeric features are z-scored; categorical ones are one-hot encoded
    (first level dropped).  The intercept is never standardized.
    """
    pieces = [np.ones((len(df), 1))]
    names = ["intercept"]

    if numeric_features:
        num = df[numeric_features].astype(float).to_numpy()
        means = num.mean(axis=0)
        stds = num.std(axis=0, ddof=0)
        stds[stds == 0] = 1.0
        num = (num - means) / stds
        pieces.append(num)
        names.extend(numeric_features)

    if categorical_features:
        for col in categorical_features:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
            pieces.append(dummies.to_numpy())
            names.extend(dummies.columns.tolist())

    X = np.hstack(pieces)
    return X, names


def load_panel(
    path: str,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Problem:
    """Load a panel CSV.

    Required columns: ``id`` plus ``revenue`` and ``cost`` cells.  Optional:
    ``profit`` (published exact profit), feature columns, and region/year
    columns used to join totals.
    """
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError("panel must contain an 'id' column")
    if "revenue" not in df.columns or "cost" not in df.columns:
        raise ValueError("panel must contain 'revenue' and 'cost' columns")
    df = df.reset_index(drop=True).copy()
    df["revenue_spec"] = df["revenue"].map(parse_cell)
    df["cost_spec"] = df["cost"].map(parse_cell)
    if "profit" in df.columns:
        df["profit"] = df["profit"].map(_as_float)
    else:
        df["profit"] = np.nan

    X, names = build_design(df, numeric_features, categorical_features)
    return Problem(df=df, X=X, feature_names=names, totals=None)


def load_totals(problem: Problem, path: str, group_cols: list[str]) -> Problem:
    """Attach a totals CSV to a :class:`Problem`.

    The totals file contains one row per group with ``total_revenue`` and
    optionally ``total_cost`` columns.  Group columns must exist in the panel.
    """
    totals = pd.read_csv(path)
    for col in group_cols + ["total_revenue"]:
        if col not in totals.columns:
            raise ValueError(f"totals file is missing column {col!r}")
    for col in group_cols:
        if col not in problem.df.columns:
            raise ValueError(f"panel is missing grouping column {col!r}")
    problem.totals = totals
    problem.totals_group_cols = group_cols
    return problem
