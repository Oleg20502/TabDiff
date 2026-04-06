"""Per-epoch KL loss multipliers for variational TabDiff training."""

from __future__ import annotations

from typing import Callable, Dict, Union

KLScheduleFn = Callable[[int], float]


class LinearWarmupKLSchedule:
    """Linearly ramp the KL coefficient from 0 to ``beta`` over ``warmup_steps`` epochs.

    Uses the same 0-based ``epoch`` index as the training loop: at ``epoch == 0`` the
    multiplier is 0; once ``epoch >= warmup_steps`` the multiplier is ``beta``.
    If ``warmup_steps <= 0``, the multiplier is always ``beta``.
    """

    def __init__(self, beta: float, warmup_steps: int):
        self.beta = float(beta)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, epoch: int) -> float:
        if self.warmup_steps <= 0:
            return self.beta
        return self.beta * min(1.0, float(epoch) / float(self.warmup_steps))


class ExponentialDecayKLSchedule:
    """Multiply the KL coefficient by ``decay_lambda`` each epoch, floored at ``beta_min``.

    At epoch ``e`` (0-based): ``coeff = max(beta_min, beta_start * decay_lambda**e)``.
    Typically ``0 < decay_lambda < 1`` so the coefficient decays toward ``beta_min``.
    """

    def __init__(self, beta_start: float, decay_lambda: float, beta_min: float):
        self.beta_start = float(beta_start)
        self.decay_lambda = float(decay_lambda)
        self.beta_min = float(beta_min)

    def __call__(self, epoch: int) -> float:
        e = max(0, int(epoch))
        v = self.beta_start * (self.decay_lambda**e)
        return max(self.beta_min, v)


def build_kl_weight_schedule(
    var_cfg: Union[Dict, None],
    use_variational: bool,
) -> KLScheduleFn:
    """Build a callable ``epoch -> kl_multiplier`` from ``[variational]`` config.

    Args:
        var_cfg: The ``variational`` section of the merged config (may be empty).
        use_variational: If False, returns a schedule that is always 0.

    Config keys (all under ``variational`` unless noted):

    - ``kl_schedule``: ``\"warmup\"`` (default) or ``\"decay\"``.
    - ``kl_weight``: Peak / initial KL multiplier (``beta`` for warmup, ``beta_start`` for decay).
    - ``kl_warmup_steps``: Warmup length in epochs (warmup schedule only).

    Decay schedule only:

    - ``kl_decay_lambda``: Multiplicative factor applied per epoch (should be in ``(0, 1)`` for decay).
    - ``kl_beta_min``: Minimum KL multiplier after decay.

    Optional table ``[variational.kl_schedule_decay]`` can provide ``kl_decay_lambda`` and
    ``kl_beta_min`` (merged over top-level keys when present).
    """
    if not use_variational:
        return lambda epoch: 0.0

    cfg = dict(var_cfg or {})
    nested = cfg.get("kl_schedule_decay")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if v is not None:
                cfg[k] = v

    name = str(cfg.get("kl_schedule", "warmup")).lower().strip()
    beta = float(cfg.get("kl_weight", 1.0))

    if name in ("warmup", "linear_warmup"):
        warmup = int(cfg.get("kl_warmup_steps", 5000))
        return LinearWarmupKLSchedule(beta, warmup)

    if name in ("decay", "exponential_decay", "kl_decay"):
        lam = float(cfg.get("kl_decay_lambda", 0.99))
        beta_min = float(cfg.get("kl_beta_min", 0.0))
        if lam <= 0.0 or lam > 1.0:
            raise ValueError(
                f"[variational] kl_decay_lambda must be in (0, 1]; got {lam!r}. "
                "Use values < 1 for decay."
            )
        if beta_min < 0.0:
            raise ValueError(f"[variational] kl_beta_min must be >= 0; got {beta_min!r}.")
        return ExponentialDecayKLSchedule(beta, lam, beta_min)

    raise ValueError(
        f"Unknown [variational].kl_schedule {name!r}; use 'warmup' or 'decay'."
    )
