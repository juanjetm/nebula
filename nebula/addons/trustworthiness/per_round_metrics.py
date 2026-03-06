# nebula/addons/trustworthiness/per_round_metrics.py
from __future__ import annotations

import asyncio
import copy
import csv
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch

from nebula.addons.functions import print_msg_box
from nebula.addons.trustworthiness.calculation import get_feature_importance_cv


def _safe_get_round(engine) -> int:
    trainer = getattr(engine, "trainer", None)
    if trainer is None:
        return -1

    try:
        return int(trainer.get_round())
    except Exception:
        return int(getattr(trainer, "round", -1))


def _get_local_test_loader(engine):
    trainer = getattr(engine, "trainer", None)
    dm = getattr(trainer, "datamodule", None)
    if dm is None:
        return None

    try:
        dm.setup(stage="test")
    except Exception:
        pass

    try:
        tdl = dm.test_dataloader()
        if isinstance(tdl, (list, tuple)) and len(tdl) > 0:
            return tdl[0]
        return tdl
    except Exception:
        return None


def _build_test_sample_min_bs(test_loader, min_bs: int = 10) -> Optional[Tuple[Any, Any]]:
    if test_loader is None:
        return None

    try:
        it = iter(test_loader)
        batch = next(it)
    except Exception:
        return None

    if not (isinstance(batch, (tuple, list)) and len(batch) >= 2):
        return None

    x, y = batch[0], batch[1]
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        return None

    if x.size(0) >= min_bs:
        return (x, y)

    xs = [x]
    ys = [y]
    cur = x.size(0)

    while cur < min_bs:
        try:
            b2 = next(it)
        except Exception:
            break
        if not (isinstance(b2, (tuple, list)) and len(b2) >= 2):
            break
        x2, y2 = b2[0], b2[1]
        if not (isinstance(x2, torch.Tensor) and isinstance(y2, torch.Tensor)):
            break
        xs.append(x2)
        ys.append(y2)
        cur += x2.size(0)

    x_cat = torch.cat(xs, dim=0)
    y_cat = torch.cat(ys, dim=0)
    return (x_cat, y_cat)


@dataclass
class PerRoundTrustMetrics:
    experiment_name: str
    participant_idx: int
    trust_dir: str
    role_label: str

    enable_print: bool = True
    enable_csv: bool = True

    _csv_path: str = field(init=False)
    _prev_acc: Optional[float] = field(default=None, init=False)
    _test_loader: Any = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def setup(self, engine) -> None:
        os.makedirs(self.trust_dir, exist_ok=True)
        self._csv_path = os.path.join(
            self.trust_dir, f"round_metrics_participant_{self.participant_idx}.csv"
        )

        if self.enable_csv and not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "round",
                    "participant",
                    "role",
                    "loss",
                    "accuracy",
                    "tw_stability",
                ])

        self._test_loader = _get_local_test_loader(engine)

    async def on_test_metrics(self, engine, loss: float, acc: float) -> None:
        async with self._lock:
            round_id = _safe_get_round(engine)

            if self._prev_acc is None:
                tw_stability = 1.0
            else:
                tw_stability = 1.0 - abs(acc - self._prev_acc)
                tw_stability = max(0.0, min(1.0, tw_stability))
            self._prev_acc = acc

            fi_cv: Optional[float] = None

            if self.enable_csv:
                with open(self._csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        round_id,
                        self.participant_idx,
                        self.role_label,
                        float(loss),
                        float(acc),
                        float(tw_stability),
                        None if fi_cv is None else float(fi_cv),
                    ])

            if self.enable_print:
                fi_txt = "NA" if fi_cv is None else f"{fi_cv:.4f}"
                print_msg_box(
                    msg=(
                        f"Round: {round_id}\n"
                        f"Loss: {loss:.4f}\n"
                        f"Accuracy: {acc:.4f}\n"
                        f"TW/Stability: {tw_stability:.4f}\n"
                    ),
                    indent=2,
                    title=f"Trustworthiness (per-round) | {self.role_label} | Participant: {self.participant_idx}",
                )
