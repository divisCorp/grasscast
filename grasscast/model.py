"""The 'taste model': TabPFN regression from window weather -> how much you'll enjoy it.

Why TabPFN: a personal outing log is tiny (dozens of rows, not thousands).
TabPFN is a pretrained tabular foundation model that does in-context learning
on small tables: no training loop, no hyper-parameter search, and it returns a
full predictive distribution, so we get honest uncertainty bands for free.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import ACTIVITY_COL, encode

MIN_OUTINGS = 8


def make_tabpfn(seed: int = 0):
    """TabPFN v2 regressor on CPU. v2 weights download without an account."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    device = os.environ.get("GRASSCAST_DEVICE", "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device=device,
            random_state=seed,
            categorical_features_indices=[ACTIVITY_COL],
            n_estimators=8,
        )


@dataclass
class TasteModel:
    activities: list[str] = field(default_factory=list)
    seed: int = 0
    _reg: object = None

    def fit(self, table: pd.DataFrame) -> "TasteModel":
        if len(table) < MIN_OUTINGS:
            raise ValueError(f"need at least {MIN_OUTINGS} logged outings, got {len(table)}")
        self.activities = sorted(table["activity"].unique())
        self._reg = make_tabpfn(self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._reg.fit(encode(table, self.activities), table["rating"].to_numpy(dtype=float))
        return self

    def predict(self, table: pd.DataFrame, great: float = 3.5) -> pd.DataFrame:
        """Expected rating, an 80% band, and P(you rate it 4 or 5 stars)."""
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self._reg.predict(encode(table, self.activities), output_type="full")
        q = np.asarray(out["quantiles"])  # default quantiles 0.1 ... 0.9
        logits = out["logits"]
        thresh = torch.full((logits.shape[0], 1), great, dtype=logits.dtype)
        p_great = 1.0 - out["criterion"].cdf(logits, thresh).squeeze(-1).detach().cpu().numpy()
        return pd.DataFrame({
            "score": np.clip(np.asarray(out["mean"]), 1, 5),
            "low": np.clip(q[0], 1, 5),
            "high": np.clip(q[-1], 1, 5),
            "p_great": np.clip(p_great, 0, 1),
        }, index=table.index)
