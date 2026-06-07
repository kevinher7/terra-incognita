"""Device-agnostic resolution — the pure half of "never hardcode the device" (PLAN §6/§9).

The selection logic is split from the *detection* deliberately:

  - :func:`resolve_device` is **pure** — given a requested device and the two
    capability bits (``has_cuda`` / ``has_mps``), it decides the concrete device. No
    torch import, so it lives in ``src/`` and is exhaustively unit-tested in lean CI.
  - The actual capability probe (``torch.cuda.is_available()`` /
    ``torch.backends.mps.is_available()``) needs the heavy ``ml`` extra, so it lives in
    the runnable training script (``scripts/train_smoke.py``) and calls this function.

That boundary is what makes local (MPS) and the GPU box (CUDA) run the *same code*: only
the resolved value differs, never a branch in the trainer.
"""

from __future__ import annotations

from terra_incognita.config import Device as ConfigDevice
from terra_incognita.obs.events import Device

__all__ = ["resolve_device", "ultralytics_device"]


def resolve_device(requested: ConfigDevice, *, has_cuda: bool, has_mps: bool) -> Device:
    """Map a requested (possibly ``auto``) device to the concrete device the run will use.

    ``auto`` is the only value we *decide*: prefer CUDA, then MPS, then CPU — the natural
    "use the best accelerator present" order. An **explicit** request is honored as written
    (the operator/GPU box asserts the hardware; if it's actually absent, Ultralytics fails
    loudly at train time — better than silently downgrading and mislabeling provenance).

    Returns a :class:`terra_incognita.obs.events.Device` (the registry enum, no ``auto``) —
    i.e. exactly the value that becomes the ``camtrap.device`` provenance/wide-event field.
    """
    if requested is ConfigDevice.auto:
        if has_cuda:
            return Device.cuda
        if has_mps:
            return Device.mps
        return Device.cpu
    # Explicit request — trust it; map the config enum onto the event enum 1:1.
    return Device(requested.value)


def ultralytics_device(device: Device) -> str | int:
    """Translate a resolved device into the value Ultralytics' ``device=`` arg expects.

    Ultralytics wants ``"cpu"`` / ``"mps"`` for those, and a **GPU index** for CUDA. We
    use ``0`` (the first GPU) — the transient single-GPU spot box (PLAN §9). Kept here,
    next to the enum it translates, so the trainer never sprinkles device-string literals.
    """
    if device is Device.cuda:
        return 0
    return device.value
