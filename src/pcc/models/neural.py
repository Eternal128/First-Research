"""Model 5 (optional): a permutation-invariant set model over players.

This model is the only one that is not told what a "nearest player" is. It
receives, for each player, a vector of raw relative quantities with respect to
the destination, and pools them with a permutation-invariant aggregator
(Deep Sets style). It therefore has the capacity to discover interaction
effects - a second defender covering the first's blind side, a goalkeeper
sweeping - that the hand-built features cannot express.

It is explicitly marked **optional** in the proposal. Including it strengthens
the upper bound on achievable calibration; excluding it costs the study
nothing essential, because the central claim concerns whether *published*
pitch-control models are calibrated, and a bespoke neural network is not one.
The minimum viable study (Section 24 of the proposal) omits it.

Requires PyTorch. The import is deferred so the rest of the package runs
without it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pcc.data.schema import ArrivalFrame
from pcc.kinematics import TTI_MODELS, LocomotionParams
from pcc.models.base import ControlModel

#: Per-player input channels. Everything is expressed relative to the
#: destination so the representation is translation-equivariant.
PLAYER_CHANNELS = 9


def player_tensor(frame: ArrivalFrame, params: LocomotionParams, tti_model: str = "bounded_accel") -> np.ndarray:
    """``(n_players, PLAYER_CHANNELS)`` representation of one arrival.

    Channels: team sign, dx, dy, distance, radial speed, tangential speed,
    time-to-point, time-to-point minus flight time, goalkeeper flag.
    """
    tti = TTI_MODELS[tti_model]
    target = frame.target
    blocks = []
    for xy, v, is_gk, sign, vmax in (
        (frame.att_xy, frame.att_v, frame.att_is_gk, +1.0, frame.att_vmax),
        (frame.def_xy, frame.def_v, frame.def_is_gk, -1.0, frame.def_vmax),
    ):
        if xy.shape[0] == 0:
            continue
        d = target[None, :] - xy
        dist = np.linalg.norm(d, axis=1)
        unit = d / np.maximum(dist[:, None], 1e-9)
        radial = np.einsum("ij,ij->i", v, unit)
        tangential = np.einsum("ij,ij->i", v, np.column_stack([-unit[:, 1], unit[:, 0]]))
        tau = np.asarray(tti(xy, v, target, params, v_max=vmax), dtype=float)
        blocks.append(
            np.column_stack(
                [
                    np.full(xy.shape[0], sign),
                    d[:, 0] / 10.0,
                    d[:, 1] / 10.0,
                    dist / 10.0,
                    radial / 5.0,
                    tangential / 5.0,
                    tau,
                    tau - frame.flight_time,
                    is_gk.astype(float),
                ]
            )
        )
    if not blocks:
        return np.zeros((0, PLAYER_CHANNELS), dtype=np.float32)
    return np.concatenate(blocks, axis=0).astype(np.float32)


class DeepSetControl(ControlModel):
    """Permutation-invariant set encoder with a global flight-time context.

    Architecture: per-player MLP ``phi``, mean-and-max pooling, concatenation
    with two global scalars (flight time, pass length), then an MLP ``rho`` to a
    single logit. Trained with binary cross-entropy, which is strictly proper.

    Calibration caveat that must be stated in any write-up: modern neural
    classifiers trained past the point of fitting the training set are known to
    be overconfident even when the loss is proper, so this model is trained with
    early stopping on a *grouped* validation split and its recalibrated variant
    is always reported alongside the raw one.
    """

    name = "M5_deepset"
    requires_fitting = True

    def __init__(
        self,
        *,
        hidden: int = 64,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int = 8,
        params: LocomotionParams | None = None,
        tti_model: str = "bounded_accel",
        random_state: int = 0,
        device: str = "cpu",
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.params = params or LocomotionParams()
        self.tti_model = tti_model
        self.random_state = random_state
        self.device = device
        self._net = None

    # -- torch plumbing ----------------------------------------------------
    @staticmethod
    def _require_torch():
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "DeepSetControl requires PyTorch. Install it, or drop Model 5 - "
                "it is optional in the study design (proposal Section 9)."
            ) from exc
        import torch

        return torch

    def _make_net(self, torch):
        nn = torch.nn
        h = self.hidden

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.phi = nn.Sequential(
                    nn.Linear(PLAYER_CHANNELS, h), nn.ReLU(),
                    nn.Linear(h, h), nn.ReLU(),
                )
                self.rho = nn.Sequential(
                    nn.Linear(2 * h + 2, h), nn.ReLU(),
                    nn.Linear(h, 1),
                )

            def forward(self, x, mask, ctx):
                e = self.phi(x) * mask[..., None]
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                mean = e.sum(dim=1) / denom
                neg_inf = torch.finfo(e.dtype).min
                mx = e.masked_fill(~mask.bool()[..., None], neg_inf).max(dim=1).values
                mx = torch.nan_to_num(mx, neginf=0.0)
                return self.rho(torch.cat([mean, mx, ctx], dim=1)).squeeze(-1)

        torch.manual_seed(self.random_state)
        return Net()

    def _encode(self, frames: Sequence[ArrivalFrame]):
        torch = self._require_torch()
        tensors = [player_tensor(f, self.params, self.tti_model) for f in frames]
        n_max = max((t.shape[0] for t in tensors), default=1)
        x = np.zeros((len(tensors), n_max, PLAYER_CHANNELS), dtype=np.float32)
        mask = np.zeros((len(tensors), n_max), dtype=np.float32)
        ctx = np.zeros((len(tensors), 2), dtype=np.float32)
        for i, (t, f) in enumerate(zip(tensors, frames)):
            x[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = 1.0
            origin = np.asarray(f.meta.get("origin", f.target), dtype=float).reshape(2)
            ctx[i] = [f.flight_time, float(np.linalg.norm(f.target - origin)) / 10.0]
        return (
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(mask).to(self.device),
            torch.from_numpy(ctx).to(self.device),
        )

    def fit(self, frames: Sequence[ArrivalFrame], y, *, sample_weight=None, val_frames=None, val_y=None):
        torch = self._require_torch()
        frames = list(frames)
        x, mask, ctx = self._encode(frames)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
        w = (
            torch.ones_like(yt)
            if sample_weight is None
            else torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)).to(self.device)
        )

        self._net = self._make_net(torch).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        val = None
        if val_frames is not None and val_y is not None:
            vx, vmask, vctx = self._encode(list(val_frames))
            vy = torch.from_numpy(np.asarray(val_y, dtype=np.float32)).to(self.device)
            val = (vx, vmask, vctx, vy)

        best, best_state, stale = np.inf, None, 0
        n = len(frames)
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            self._net.train()
            for idx in np.array_split(rng.permutation(n), max(1, n // self.batch_size)):
                if idx.size == 0:
                    continue
                opt.zero_grad()
                logits = self._net(x[idx], mask[idx], ctx[idx])
                loss = (loss_fn(logits, yt[idx]) * w[idx]).mean()
                loss.backward()
                opt.step()

            if val is not None:
                self._net.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(self._net(val[0], val[1], val[2]), val[3]).mean())
                if vloss < best - 1e-5:
                    best, stale = vloss, 0
                    best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, frames: Sequence[ArrivalFrame]) -> np.ndarray:
        torch = self._require_torch()
        if self._net is None:
            raise RuntimeError(f"{self.name} must be fitted before prediction")
        self._net.eval()
        x, mask, ctx = self._encode(list(frames))
        with torch.no_grad():
            return torch.sigmoid(self._net(x, mask, ctx)).cpu().numpy().astype(float)

    def control(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        if targets is None:
            return self.predict([frame])
        stubs = []
        for t in np.atleast_2d(targets):
            stubs.append(
                ArrivalFrame(
                    att_xy=frame.att_xy, att_v=frame.att_v, def_xy=frame.def_xy, def_v=frame.def_v,
                    target=t, flight_time=float(self._flight_times(frame, t[None, :])[0]),
                    att_is_gk=frame.att_is_gk, def_is_gk=frame.def_is_gk,
                    att_vmax=frame.att_vmax, def_vmax=frame.def_vmax, meta=frame.meta,
                )
            )
        return self.predict(stubs)
