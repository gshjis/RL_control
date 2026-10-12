from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Raw telemetry from C++ `co_sensor` / `get_telemetry`
# ---------------------------------------------------------------------------
class Telemetry(NamedTuple):
    """8 наблюдений от C++ сенсора (порядок важен!)."""
    x: float                # координата тележки
    cosθ1: float
    sinθ1: float
    cosθ2: float
    sinθ2: float
    dx: float               # ẋ — скорость тележки
    dθ1: float              # θ̇1 — угловая скорость 1-го звена
    dθ2: float              # θ̇2 — угловая скорость 2-го звена


# ---------------------------------------------------------------------------
# FeatureConfig — единственный источник правды, что видит НС (и прошивка)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Конфигурация набора признаков для политики и прошивки Arduino.

    n_links      : 1 | 2   (число звеньев маятника)
    position_controlled : bool
        True — в вектор добавляется ошибка положенияテレжки  e_x = x − x_target.
        False — swing‑up режим, ошибки не добавляем.
    """

    n_links: int = field(default=1, metadata={"valid": "1 | 2"})
    position_controlled: bool = field(default=False, metadata={"valid": "bool"})

    # ---- invariants --------------------------------------------------------
    @property
    def n_links(self) -> int:
        return self._n_links

    @n_links.setter  # type: ignore[method-defined]
    def n_links(self, value: int) -> None:
        if value not in (1, 2):
            raise ValueError("n_links must be 1 or 2")
        self._n_links = value

    # ---- размерность вектора признаков ------------------------------------
    @property
    def dim(self) -> int:
        """Number of features the NN input layer expects."""
        base = 3 * self.n_links + 1  # sin, cos, θ̇ for each link + ẋ
        if self.position_controlled:
            base += 1  # e_x
        return base

    # ---- human-readable description ----------------------------------------
    def __repr__(self) -> str:
        ctrl = "control" if self.position_controlled else "swing‑up"
        return f"FeatureConfig(n_links={self.n_links}, mode={ctrl}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Helpers – error calculation
# ---------------------------------------------------------------------------
def _e_x(x: float, x_target: float) -> float:
    """Cart‑position error used when position_controlled=True."""
    return float(x - x_target)


# ---------------------------------------------------------------------------
# Core builder – produces the exact vector the policy expects
# ---------------------------------------------------------------------------
def build_features(
    tel: Telemetry,
    *,
    cfg: FeatureConfig,
    x_target: float | None = None,
) -> np.ndarray:
    """Build the feature vector the NN (and Arduino firmware) expects.

    Parameters
    ----------
    tel : Telemetry
        Raw 8‑component observation from C++.
    cfg : FeatureConfig
        Configuration deciding which subset to emit.
    x_target : float, optional
        Required iff ``cfg.position_controlled`` is True.  If None and
        position_controlled=True a ``ValueError`` is raised.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(cfg.dim,)``, dtype float64.
    """
    if cfg.position_controlled and x_target is None:
        raise ValueError("x_target must be provided when position_controlled=True")

    # --- extract from telemetry (names match the C++ struct) ----------------
    sinθ1, cosθ1 = tel.sinθ1, tel.cosθ1
    sinθ2, cosθ2 = tel.sinθ2, tel.cosθ2
    θ̇1, θ̇2     = tel.dθ1, tel.dθ2
    x, ẋ         = tel.x, tel.dx

    # --- build the list in the canonical order --------------------------------
    #   [sinθ1, cosθ1, θ̇1, ẋ, (e_x), cosθ2, sinθ2, θ̇2]
    features: list[float] = [sinθ1, cosθ1, θ̇1, ẋ]

    if cfg.position_controlled:
        if x_target is None:
            raise ValueError("x_target must be provided when position_controlled=True")
        features.append(_e_x(x, x_target))

    if cfg.n_links == 2:
        features += [cosθ2, sinθ2, θ̇2]

    # At this point len(features) must equal cfg.dim
    assert len(features) == cfg.dim, (
        f"Feature length {len(features)} != dim {cfg.dim}; "
        f"cfg.n_links={cfg.n_links}, position_controlled={cfg.position_controlled}"
    )

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Unit tests – quick smoke check that dim / values are correct
# ---------------------------------------------------------------------------
def _make_tel() -> Telemetry:
    """Convenient fake telemetry (all zeros except non‑zero ẋ)."""
    return Telemetry(
        x=0.2,
        cosθ1=0.8,
        sinθ1=0.6,
        cosθ2=0.1,
        sinθ2=-0.2,
        dx=0.05,
        dθ1=0.3,
        dθ2=-0.1,
    )


def test_feature_dims() -> None:
    """Every FeatureConfig combo yields the dim promised by the table."""
    configs: list[FeatureConfig] = [
        FeatureConfig(n_links=1, position_controlled=False),   # swing‑up 1‑link: 4
        FeatureConfig(n_links=1, position_controlled=True),   # control 1‑link: 5
        FeatureConfig(n_links=2, position_controlled=False),  # swing‑up 2‑link: 7
        FeatureConfig(n_links=2, position_controlled=True),   # control 2‑link: 8
    ]
    expected = [4, 5, 7, 8]
    for cfg, exp in zip(configs, expected):
        tel = _make_tel()
        # supply x_target only when required
        xt = 0.0 if cfg.position_controlled else None
        feats = build_features(tel, cfg=cfg, x_target=xt)
        assert feats.shape == (exp,), f"cfg={cfg}: shape {feats.shape} != {exp}"
    print("All dim tests passed.")


def test_feature_values() -> None:
    """Verify that the canonical order and e_x are correct."""
    tel = _make_tel()
    # 1‑link, control, x_target = 0.0 → e_x = 0.2 - 0.0 = 0.2
    cfg = FeatureConfig(n_links=1, position_controlled=True)
    feats = build_features(tel, cfg=cfg, x_target=0.0)

    # manually check first 4 entries (sin, cos, θ̇, ẋ)
    assert feats[0] == 0.6, f"sinθ1 expected 0.6 got {feats[0]}"
    assert feats[1] == 0.8, f"cosθ1 expected 0.8 got {feats[1]}"
    assert feats[2] == 0.3, f"θ̇1 expected 0.3 got {feats[2]}"
    assert feats[3] == 0.05, f"ẋ expected 0.05 got {feats[3]}"
    # e_x should be 0.2
    assert feats[4] == 0.2, f"e_x expected 0.2 got {feats[4]}"

    # 2‑link, control
    cfg2 = FeatureConfig(n_links=2, position_controlled=True)
    feats2 = build_features(tel, cfg=cfg2, x_target=0.0)
    # first 5 are identical to 1‑link control
    assert np.allclose(feats2[:5], feats[:5])
    # then cosθ2, sinθ2, θ̇2
    assert feats2[5] == 0.1, f"cosθ2 expected 0.1 got {feats2[5]}"
    assert feats2[6] == -0.2, f"sinθ2 expected -0.2 got {feats2[6]}"
    assert feats2[7] == -0.1, f"θ̇2 expected -0.1 got {feats2[7]}"

    # 2‑link, swing‑up (no control)
    cfg3 = FeatureConfig(n_links=2, position_controlled=False)
    feats3 = build_features(tel, cfg=cfg3, x_target=None)
    assert np.allclose(feats3[:4], feats[:4])          # first 4 shared
    assert feats3[4] != feats[4], "e_x should NOT be present when control=False"
    assert feats3[4:7] == np.array([0.1, -0.2, -0.1]), (
        f"swing‑up 2‑link middle entries wrong: {feats3[4:7]}"
    )

    print("All value tests passed.")


if __name__ == "__main__":
    test_feature_dims()
    test_feature_values()