"""
patch_clip_fp8.py
~~~~~~~~~~~~~~~~~
Replaces nn.Linear weight storage with FP8 buffers so that large text-encoder
weights occupy less VRAM while remaining transparently usable at runtime.

Design notes
------------
* Works when the CLIP object is a ComfyUI wrapper (not a bare nn.Module).
* If the running PyTorch build does not expose ``torch.float8_e4m3fn`` the
  module imports successfully but ``FP8_AVAILABLE`` is False.  All public
  functions detect this and return the model unchanged, giving the same result
  as a standard FP16 load.
* Weight surgery is in-place: the original nn.Linear gains an
  ``_fp8_weight_storage`` attribute and a patched ``forward`` method; its
  ``weight`` Parameter is removed and replaced by a non-persistent buffer so
  that ComfyUI serialisation/inspection code that iterates ``_parameters``
  sees no weight entry (expected — the buffer lives in ``_buffers``).
"""

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# ---------------------------------------------------------------------------
# FP8 availability gate
# ---------------------------------------------------------------------------

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_AVAILABLE = FP8_DTYPE is not None

if not FP8_AVAILABLE:
    import warnings
    warnings.warn(
        "comfy_fp8: torch.float8_e4m3fn is not available in this PyTorch build "
        "(requires PyTorch >= 2.1 with CUDA/ROCm FP8 support). "
        "FP8 weight storage will be skipped; weights stay in FP16.",
        RuntimeWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# FP8 weight storage helper
# ---------------------------------------------------------------------------

class _FP8WeightStorage:
    """
    Stores one Linear weight as an FP8 buffer on the target device and
    materialises it back to a higher-precision dtype on demand.

    The buffer is registered as *non-persistent* so it is excluded from
    ``state_dict()`` / serialisation — the canonical source of truth
    remains the original checkpoint.

    Parameters
    ----------
    module : nn.Module
        The Linear layer that owns this storage object.
    name : str
        Attribute name of the weight being replaced (typically ``"weight"``).
    weight_tensor : torch.Tensor
        The original weight in its checkpoint dtype.
    device : torch.device
        Device on which the FP8 buffer should live.
    """

    def __init__(self, module: nn.Module, name: str, weight_tensor: torch.Tensor, device: torch.device):
        self.module = module
        self.buf_name = f"__fp8_{name}"
        w_fp8 = weight_tensor.detach().to(device).to(FP8_DTYPE)
        # non-persistent so it is not included in state_dict() exports
        module.register_buffer(self.buf_name, w_fp8, persistent=False)

    def materialize(self, compute_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Return the weight cast to *compute_dtype* (default: float16)."""
        return getattr(self.module, self.buf_name).to(compute_dtype)


# ---------------------------------------------------------------------------
# Core weight-surgery pass
# ---------------------------------------------------------------------------

def _replace_linears_with_fp8_storage(
    module: nn.Module,
    compute_dtype: torch.dtype = torch.float16,
    device: torch.device = None,
) -> nn.Module:
    """
    Recursively walk *module* and convert every ``nn.Linear`` weight to FP8
    storage in-place.

    Each converted layer:
    * Has its ``weight`` Parameter removed from ``_parameters``.
    * Gains a non-persistent ``__fp8_weight`` buffer in ``_buffers``.
    * Gains an ``_fp8_weight_storage`` attribute (an ``_FP8WeightStorage``).
    * Has its ``forward`` method replaced with one that materialises the FP8
      buffer and casts it to match the input dtype before calling
      ``F.linear``.

    The inner ``_linear_forward`` is defined once outside the loop and bound
    per-instance with ``types.MethodType`` — this avoids both the classic
    Python loop-closure bug *and* repeated function object creation.
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define the replacement forward once; MethodType binding supplies `self`
    # for each individual layer so there is no closure over the loop variable.
    def _linear_forward(self, input: torch.Tensor) -> torch.Tensor:
        # Materialise the FP8 buffer to the helper's default compute dtype.
        w = self._fp8_weight_storage.materialize()

        # Cast weight (and bias if present) to match the incoming activation
        # dtype, e.g. bfloat16 if the rest of the graph is running in bf16.
        input_dtype = input.dtype
        if w.dtype != input_dtype:
            w = w.to(input_dtype)

        b = getattr(self, "bias", None)
        if b is not None and b.dtype != input_dtype:
            # Cast a view for this call only — do not mutate the stored bias.
            b = b.to(input_dtype)

        return F.linear(input, w, b)

    for name, child in list(module.named_children()):
        # Recurse depth-first so children are processed before we inspect
        # the current level (order doesn't matter functionally, but
        # depth-first keeps the traversal predictable).
        _replace_linears_with_fp8_storage(child, compute_dtype=compute_dtype, device=device)

        if isinstance(child, nn.Linear):
            # Idempotency guard: skip layers already converted in a previous
            # run.  ComfyUI may return a cached CLIP object on subsequent
            # executions; without this check _FP8WeightStorage.__init__ would
            # allocate a new FP8 buffer and register_buffer would overwrite the
            # old one, causing a brief period of double-buffering per layer
            # (old buffer not yet GC'd + new buffer) that inflates reported
            # VRAM on runs 2+.
            if hasattr(child, "_fp8_weight_storage"):
                continue

            weight = child.weight.detach()
            bias = child.bias.detach() if child.bias is not None else None

            # Build FP8 storage and register the buffer on the child module.
            storage = _FP8WeightStorage(child, "weight", weight, device)

            # Remove the original weight Parameter so nothing accidentally
            # reads a stale high-precision copy.  Note: after this removal
            # ``child.weight`` will raise AttributeError — that is intentional.
            # ComfyUI model_patcher is patched in load_clip_fp8.py to handle
            # this via the _fp8_weight_storage fallback path.
            try:
                del child._parameters["weight"]
            except KeyError:
                pass

            if bias is not None:
                child._parameters["bias"] = nn.Parameter(bias.to(compute_dtype))

            child._fp8_weight_storage = storage

            # Bind the shared forward function to this specific child instance.
            child.forward = types.MethodType(_linear_forward, child)

    return module


# ---------------------------------------------------------------------------
# BFS wrapper-discovery helper
# ---------------------------------------------------------------------------

def _find_first_module(obj, max_nodes: int = 256):
    """
    Breadth-first search for the first ``nn.Module`` reachable from *obj*.

    ComfyUI's CLIP object is a multi-layer wrapper; this function navigates
    those layers without requiring knowledge of the specific wrapper class.

    Returns
    -------
    (module, path) : (nn.Module | None, list[str] | None)
        *module* is the first ``nn.Module`` found; *path* is the list of
        attribute names traversed to reach it from *obj*.
        Returns ``(None, None)`` if no module is found within *max_nodes*
        examined nodes.
    """
    if isinstance(obj, nn.Module):
        return obj, []

    visited = set()
    queue = deque([(obj, [])])
    nodes_examined = 0

    while queue and nodes_examined < max_nodes:
        nodes_examined += 1
        current, path = queue.popleft()

        if id(current) in visited:
            continue
        visited.add(id(current))

        if isinstance(current, nn.Module):
            return current, path

        # Check common ComfyUI/Transformers attribute names first for speed.
        for attr in ("model", "text_encoder", "transformer", "encoder",
                     "clip", "vision_model", "text_model"):
            try:
                child = getattr(current, attr, None)
            except Exception:
                child = None
            if child is None:
                continue
            if isinstance(child, nn.Module):
                return child, path + [attr]
            queue.append((child, path + [attr]))

        # Generic attribute scan as a fallback.
        try:
            for attr_name in dir(current):
                if attr_name.startswith("_"):
                    continue
                try:
                    child = getattr(current, attr_name)
                except Exception:
                    continue
                if child is None:
                    continue
                if isinstance(child, nn.Module):
                    return child, path + [attr_name]
                if isinstance(child, (list, tuple)):
                    for i, item in enumerate(child):
                        if isinstance(item, nn.Module):
                            return item, path + [attr_name, str(i)]
                        queue.append((item, path + [attr_name, str(i)]))
                elif isinstance(child, dict):
                    for k, v in child.items():
                        if isinstance(v, nn.Module):
                            return v, path + [attr_name, str(k)]
                        queue.append((v, path + [attr_name, str(k)]))
                else:
                    queue.append((child, path + [attr_name]))
        except Exception:
            # Some objects raise on dir() or getattr(); skip and continue.
            pass

    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_fp8_storage_to_module_or_wrapper(obj, compute_dtype: torch.dtype = torch.float16, device=None):
    """
    Apply FP8 weight storage to every ``nn.Linear`` inside *obj*.

    Accepts either a bare ``nn.Module`` or a ComfyUI wrapper object that
    contains one.  The object is mutated in place and returned.

    If FP8 is not available in the current PyTorch build (``FP8_AVAILABLE``
    is False) this function is a no-op: *obj* is returned unchanged, giving
    behaviour equivalent to a standard FP16 load.

    Parameters
    ----------
    obj : nn.Module or wrapper object
        The model (or wrapper) whose Linear weights should be converted.
    compute_dtype : torch.dtype
        The dtype used when materialising FP8 weights at inference time.
        Defaults to ``torch.float16``.
    device : torch.device or None
        Device on which FP8 buffers should be allocated.  Defaults to CUDA
        if available, otherwise CPU.
    """
    if not FP8_AVAILABLE:
        # Graceful no-op: caller gets an unmodified model and normal FP16 behaviour.
        return obj

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if isinstance(obj, nn.Module):
        _replace_linears_with_fp8_storage(obj, compute_dtype=compute_dtype, device=device)
        return obj

    # Unwrap the ComfyUI CLIP wrapper to find the underlying nn.Module.
    module, path = _find_first_module(obj)
    if module is None:
        raise RuntimeError(
            "apply_fp8_storage_to_module_or_wrapper: no nn.Module found inside "
            "the provided object. FP8 storage could not be applied."
        )

    _replace_linears_with_fp8_storage(module, compute_dtype=compute_dtype, device=device)

    # Return the original wrapper so callers continue to use the same reference.
    return obj


# Backwards-compatible alias used by load_clip_fp8.py.
def apply_fp8_storage_to_module(model, compute_dtype: torch.dtype = torch.float16, device=None):
    """Alias for :func:`apply_fp8_storage_to_module_or_wrapper`."""
    return apply_fp8_storage_to_module_or_wrapper(model, compute_dtype=compute_dtype, device=device)