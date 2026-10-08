"""How well does the taste model know you? Cross-validated against simple baselines."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from .features import TIME_FEATURES, WEATHER_FEATURES
from .model import TasteModel


def _sk_xy(table):
    return table[WEATHER_FEATURES + TIME_FEATURES + ["activity"]], table["rating"].to_numpy(float)


def _ridge():
    pre = ColumnTransformer([("num", StandardScaler(), WEATHER_FEATURES + TIME_FEATURES),
                             ("act", OneHotEncoder(handle_unknown="ignore"), ["activity"])])
    return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 20)))


def _forest(seed):
    pre = ColumnTransformer([("num", "passthrough", WEATHER_FEATURES + TIME_FEATURES),
                             ("act", OneHotEncoder(handle_unknown="ignore"), ["activity"])])
    return make_pipeline(pre, RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed))


def predict_all(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
    out = {}
    out["always guess your average"] = np.full(len(test), train["rating"].mean())
    act_mean = train.groupby("activity")["rating"].mean()
    out["average per activity"] = test["activity"].map(act_mean).fillna(train["rating"].mean()).to_numpy()
    Xtr, ytr = _sk_xy(train)
    Xte, _ = _sk_xy(test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["ridge regression"] = _ridge().fit(Xtr, ytr).predict(Xte)
        out["random forest"] = _forest(seed).fit(Xtr, ytr).predict(Xte)
    out["TabPFN (grasscast)"] = TasteModel(seed=seed).fit(train).predict(test)["score"].to_numpy()
    return out


def _spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def cross_validate(table: pd.DataFrame, folds: int = 5, repeats: int = 2) -> pd.DataFrame:
    preds: dict[str, list] = {}
    truth = []
    for rep in range(repeats):
        for tr, te in KFold(folds, shuffle=True, random_state=rep).split(table):
            p = predict_all(table.iloc[tr], table.iloc[te], seed=rep)
            for k, v in p.items():
                preds.setdefault(k, []).append(v)
            truth.append(table.iloc[te]["rating"].to_numpy())
    y = np.concatenate(truth)
    rows = []
    for k, v in preds.items():
        v = np.concatenate(v)
        rows.append({"model": k, "MAE (stars)": np.abs(v - y).mean(), "rank corr": _spearman(v, y)})
    return pd.DataFrame(rows).sort_values("MAE (stars)").reset_index(drop=True)


def learning_curve(table: pd.DataFrame, sizes=(10, 15, 20, 30, 40), test_size: int = 20, repeats: int = 3) -> pd.DataFrame:
    """MAE on held-out outings as the log grows: how many outings until it knows you?"""
    rows = []
    for rep in range(repeats):
        rng = np.random.default_rng(100 + rep)
        idx = rng.permutation(len(table))
        test, pool = table.iloc[idx[:test_size]], table.iloc[idx[test_size:]]
        for n in sizes:
            if n > len(pool):
                continue
            p = predict_all(pool.iloc[:n], test, seed=rep)
            y = test["rating"].to_numpy()
            for k in ("average per activity", "random forest", "TabPFN (grasscast)"):
                rows.append({"outings": n, "model": k, "mae": np.abs(p[k] - y).mean()})
    df = pd.DataFrame(rows)
    return df.pivot_table(index="outings", columns="model", values="mae").round(2)
