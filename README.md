# comfyui-fp8-load-clip

A ComfyUI custom node that replaces the stock **Load CLIP** node with one that stores text encoder weights as FP8 buffers on the GPU, nearly halving persistent VRAM consumption for large models while preserving full inference correctness.

---

## The problem

When ComfyUI loads a text encoder checkpoint it stores every `nn.Linear` weight as a standard `torch.float16` `Parameter`. For large models such as Qwen3 8B this produces a persistent VRAM footprint that exceeds what most consumer GPUs can hold alongside other models in the same workflow. The stock node handles this through lowvram offloading, but offloading imposes latency and reduces the memory available to the diffusion model and VAE.

**Stock Load CLIP node — Qwen3 8B FP8 checkpoint:**
```
loaded partially; 14559.11 MB usable, 14334.59 MB loaded,
1288.00 MB offloaded, 224.00 MB buffer reserved, lowvram patches: 0
```

The full FP16 weight set occupies ~14.3 GB, exceeding available VRAM and forcing partial offloading to system RAM.

---

## The solution

`float8_e4m3fn` (FP8) uses one byte per value versus two for FP16. Storing weights in FP8 therefore cuts the persistent weight footprint roughly in half. At inference time each weight tensor is materialised back to the activation dtype (typically FP16 or BF16) immediately before the matrix multiply and discarded afterward, so the compute path is identical to a standard FP16 model.

**Load CLIP FP8 node — same checkpoint:**
```
loaded completely; 7574.43 MB usable, 8405.09 MB loaded, full load: True
```

The model loads completely into VRAM at ~8.4 GB with no offloading, freeing the remainder for other models in the workflow.

---

## How it works

### 1. Weight surgery (`patch_clip_fp8.py`)

After the checkpoint is loaded by `comfy.sd.load_clip`, every `nn.Linear` in the text encoder is visited. For each layer:

- The `weight` entry is removed from `module._parameters`.
- The weight tensor is cast to `torch.float8_e4m3fn` and registered as a non-persistent buffer (`module._buffers`) via an `_FP8WeightStorage` helper object.
- The layer's `forward` method is replaced with one that materialises the FP8 buffer to the incoming activation dtype before calling `F.linear`, then discards the upcast copy.

The surgery is in-place and idempotent: a per-layer `_fp8_weight_storage` guard prevents double-conversion if ComfyUI returns a cached model object on a subsequent run.

### 2. Model patcher compatibility (`load_clip_fp8.py`)

ComfyUI's `ModelPatcher.get_key_weight` expects weights to be accessible as regular `Parameter` attributes. Since the surgery removes them from `_parameters`, a one-time monkeypatch is applied to `comfy.model_patcher.get_key_weight` that falls back to `_fp8_weight_storage.materialize()` whenever the normal attribute lookup fails. The patch is process-scoped, idempotent, and fully backwards-compatible with unpatched layers.

### 3. Memory accounting (`load_clip_fp8.py`)

ComfyUI tracks loaded weight memory through `comfy.model_management.module_size`, which sums `named_parameters()` on each submodule. Because the FP8 weights live in `_buffers` rather than `_parameters`, the original function reports only biases — roughly 1.2 GB — causing the scheduler to treat an 8.4 GB model as nearly empty. Two corrections are applied:

- **`module_size` monkeypatch**: `comfy.model_management.module_size` is replaced with a version that sums both `parameters(recurse=False)` and `buffers(recurse=False)`, using `.nbytes` directly so that FP8's one-byte-per-element storage is counted correctly. This fixes both the console display and the VRAM scheduler's eviction and offloading decisions.
- **`model_size` method patch**: `patcher.model.model_size` is replaced with a lambda returning the correctly computed byte total, fixing the scheduling path that calls `model_size()` when determining whether a full load is feasible.

Both patches are process-scoped and idempotent.

### 4. Graceful fallback

`FP8_AVAILABLE` is set at import time by probing for `torch.float8_e4m3fn`. If the attribute is absent (PyTorch < 2.1 or a build without FP8 support) all patches are skipped and the node behaves identically to the stock Load CLIP node.

---

## Requirements

- ComfyUI (recent stable release)
- PyTorch ≥ 2.1 with a CUDA or ROCm build that exposes `torch.float8_e4m3fn`
- A GPU and driver compatible with the above PyTorch build

---

## Installation

Place the folder inside ComfyUI's `custom_nodes` directory:

```
<ComfyUI root>/custom_nodes/comfyui-fp8-load-clip/
  ├── __init__.py
  ├── patch_clip_fp8.py
  └── load_clip_fp8.py
```

Restart ComfyUI. The node is discovered automatically.

---

## Usage

The node appears in the node browser as **Load CLIP FP8** under the category **custom/loaders**.

| Input | Type | Description |
|---|---|---|
| `clip_name` | string | Checkpoint filename from the `text_encoders` folder |
| `clip_type_name` | enum | Architecture variant (stable\_diffusion, flux2, wan, hidream, …) |
| `device` | enum | `default` uses ComfyUI's normal placement; `cpu` forces CPU load and offload |

The output is a standard `CLIP` object identical in type to the stock node's output and compatible with all downstream nodes.

---

## Verification

After loading a model, the following can be run in the ComfyUI Python console to confirm FP8 storage is active:

```python
from custom_nodes.comfyui_fp8_load_clip.patch_clip_fp8 import _find_first_module
import torch

module, path = _find_first_module(clip)
print("module path:", path)

for n, m in module.named_modules():
    if isinstance(m, torch.nn.Linear) and hasattr(m, "_fp8_weight_storage"):
        buf = getattr(m, m._fp8_weight_storage.buf_name)
        print(f"patched linear : {n}")
        print(f"buffer dtype   : {buf.dtype}")       # expect torch.float8_e4m3fn
        print(f"buffer device  : {buf.device}")      # expect cuda:0
        break

print(f"cuda allocated : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

---

## Troubleshooting

**`torch.float8_e4m3fn` not available**
Upgrade to PyTorch ≥ 2.1 compiled with FP8 support. The node falls back to standard FP16 loading automatically and will print a `RuntimeWarning` at import time.

**Dtype or device mismatch during inference**
The node materialises FP8 weights to match the incoming activation dtype at each forward pass. Errors here typically indicate that the model and its inputs are on different devices. Confirm both are on `cuda:0`.

**ComfyUI internals differ significantly**
The monkeypatches target `comfy.model_patcher.get_key_weight` and `comfy.model_management.module_size`. If ComfyUI has been updated and these functions have changed signature or semantics, the patches may need adjustment. File an issue with your ComfyUI version and the full traceback.

---

## Compatibility notes

- The node does not modify any ComfyUI core files; all changes are applied as runtime monkeypatches scoped to the current process.
- LoRA and other model patches applied via ComfyUI's standard patching system are unaffected; the `get_key_weight` patch preserves all original code paths for unpatched layers.
- The node is compatible with any checkpoint that can be loaded by the stock Load CLIP node. The FP8 conversion is applied post-load regardless of whether the checkpoint itself is stored in FP8, FP16, or BF16 on disk.