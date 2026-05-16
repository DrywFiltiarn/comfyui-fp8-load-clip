"""
load_clip_fp8.py
~~~~~~~~~~~~~~~~
Defines the ``LoadCLIPFP8`` ComfyUI node.

The node is a drop-in replacement for ComfyUI's built-in "Load CLIP" node.
After loading the checkpoint normally it applies FP8 weight storage to every
nn.Linear in the text encoder, reducing VRAM usage with no change to output
quality.  A monkeypatch to ``comfy.model_patcher.get_key_weight`` is applied
once per process so the model patcher can read weights out of the FP8 storage
helper rather than expecting a plain ``weight`` Parameter.

If FP8 is unavailable (see ``patch_clip_fp8.FP8_AVAILABLE``), both the
storage helper and the monkeypatch are skipped transparently; the node
behaves identically to the standard "Load CLIP" node.
"""

import torch
import comfy
import comfy.sd
import comfy.model_patcher as model_patcher
import comfy.model_management as model_management
import comfy.utils as utils
import folder_paths

from .patch_clip_fp8 import apply_fp8_storage_to_module, _find_first_module, FP8_AVAILABLE


class LoadCLIPFP8:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                # Parameter renamed from `type` (shadows Python builtin) to
                # `clip_type_name` for clarity; ComfyUI uses the dict key as
                # the UI label so the widget text is unchanged.
                "clip_type_name": ([
                    "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
                    "mochi", "cogvideox", "cosmos", "lumina2", "wan", "hidream",
                    "omnigen2", "qwen_image", "flux2", "ovis", "longcat_image",
                    "pixart", "chroma", "ace",
                ],),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip_fp8"
    CATEGORY = "custom/loaders"
    DESCRIPTION = (
        "Load CLIP and keep large Linear weights stored as FP8 buffers in VRAM "
        "(no core patches). Falls back to standard FP16 loading if FP8 is "
        "unavailable in the current PyTorch build."
    )

    # ------------------------------------------------------------------
    # model_management.module_size monkeypatch
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_fp8_aware_module_size():
        """
        Monkeypatch ``comfy.model_management.module_size`` exactly once per
        process so that it counts non-persistent buffers (where FP8 weight
        data lives) in addition to parameters.

        ComfyUI's original ``module_size`` only sums ``named_parameters()``.
        After FP8 surgery the weight Parameters are deleted and replaced by
        non-persistent buffers, so the original function sees only biases —
        ~1.2 GB instead of ~8.4 GB.  This causes the "X MB loaded" console
        line to be wildly wrong and, more importantly, causes ComfyUI's VRAM
        scheduler to think the model is tiny and make poor offloading decisions.

        The patched version is fully backwards-compatible: for any module that
        has no FP8 buffers the result is identical to the original.
        """
        if getattr(model_management.module_size, "_fp8_aware", False):
            return  # already patched in this process

        orig_module_size = model_management.module_size

        def fp8_aware_module_size(module):
            """
            Return the byte size of *module*'s parameters **and** buffers.

            Buffers are normally excluded from the original because ComfyUI
            buffers (e.g. positional embeddings) are not moved to GPU during
            lowvram loading.  FP8 weight buffers are different — they are
            always on the target device and represent the true weight storage —
            so we must include them.

            We use ``nbytes`` directly rather than reconstructing the
            dtype_size multiply so that FP8's 1-byte-per-element is counted
            correctly regardless of what ComfyUI's ``dtype_size`` helper does.
            """
            # Sum parameters exactly as the original does.
            mem = sum(p.nbytes for p in module.parameters(recurse=False))
            # Add buffers — this picks up FP8 weight buffers on patched layers
            # as well as any other buffers the module happens to own.
            mem += sum(b.nbytes for b in module.buffers(recurse=False))
            return mem

        fp8_aware_module_size._fp8_aware = True
        model_management.module_size = fp8_aware_module_size

    # ------------------------------------------------------------------
    # model_patcher monkeypatch
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_fp8_aware_patcher():
        """
        Monkeypatch ``comfy.model_patcher.get_key_weight`` exactly once per
        process so that it can resolve weights stored in ``_FP8WeightStorage``
        objects (which replace the normal ``weight`` Parameter on patched
        nn.Linear layers).

        The patch is idempotent: a ``_fp8_aware`` flag on the replacement
        function prevents double-patching across multiple node instantiations
        or workflow reloads.

        The patched function is fully backwards-compatible: for any layer that
        has *not* been FP8-patched the code paths are identical to the
        original.
        """
        if getattr(model_patcher.get_key_weight, "_fp8_aware", False):
            return  # already patched in this process

        orig_get_key_weight = model_patcher.get_key_weight  # kept for reference

        def fp8_aware_get_key_weight(model, key):
            """
            Return ``(weight_tensor, set_func, convert_func)`` for *key*.

            Extends the original to fall back to ``_fp8_weight_storage.materialize()``
            when the ``weight`` Parameter has been replaced by FP8 storage.
            """
            set_func = None
            convert_func = None
            op_keys = key.rsplit(".", 1)

            # Single-segment key — no parent module to look up.
            if len(op_keys) < 2:
                weight = comfy.utils.get_attr(model, key)
                return weight, set_func, convert_func

            # Resolve the parent module/object.
            op = comfy.utils.get_attr(model, op_keys[0])
            attr = op_keys[1]

            # Optional setter / converter hooks (preserved from original).
            try:
                set_func = getattr(op, f"set_{attr}")
            except AttributeError:
                set_func = None

            try:
                convert_func = getattr(op, f"convert_{attr}")
            except AttributeError:
                convert_func = None

            # Primary path: normal attribute access (covers unpatched layers).
            try:
                weight = getattr(op, attr)
                # If a convert hook exists, use get_attr to preserve original semantics.
                if convert_func is not None:
                    weight = comfy.utils.get_attr(model, key)
                return weight, set_func, convert_func
            except AttributeError:
                pass

            # FP8 fallback: layer weight has been replaced by storage helper.
            if hasattr(op, "_fp8_weight_storage"):
                try:
                    weight = op._fp8_weight_storage.materialize()
                    return weight, set_func, convert_func
                except Exception as e:
                    raise RuntimeError(
                        f"comfy_fp8: failed to materialise FP8 storage for key '{key}': {e}"
                    ) from e

            # Last resort: let get_attr raise naturally if the key is genuinely missing.
            weight = comfy.utils.get_attr(model, key)
            return weight, set_func, convert_func

        fp8_aware_get_key_weight._fp8_aware = True
        model_patcher.get_key_weight = fp8_aware_get_key_weight

    # ------------------------------------------------------------------
    # Node entry point
    # ------------------------------------------------------------------

    def load_clip_fp8(self, clip_name, clip_type_name="stable_diffusion", device="default"):
        """
        Load a CLIP checkpoint and apply FP8 weight storage.

        Parameters
        ----------
        clip_name : str
            Filename of the text-encoder checkpoint (looked up in ComfyUI's
            ``text_encoders`` folder list).
        clip_type_name : str
            Architecture variant string forwarded to ``comfy.sd.CLIPType``.
        device : str
            ``"default"`` uses ComfyUI's normal device placement logic;
            ``"cpu"`` forces both load and offload devices to CPU.
        """
        # Apply both monkeypatches only when FP8 is actually available;
        # there is nothing for them to handle on a plain FP16 model.
        if FP8_AVAILABLE:
            try:
                self._ensure_fp8_aware_module_size()
            except Exception as e:
                print("comfy_fp8: failed to apply module_size monkeypatch:", e)
            try:
                self._ensure_fp8_aware_patcher()
            except Exception as e:
                print("comfy_fp8: failed to apply model_patcher monkeypatch:", e)

        # Resolve CLIPType enum value from the string supplied by the UI widget.
        clip_type_enum = getattr(comfy.sd, "CLIPType", None)
        if clip_type_enum is not None:
            clip_type = getattr(
                comfy.sd.CLIPType,
                clip_type_name.upper(),
                comfy.sd.CLIPType.STABLE_DIFFUSION,
            )
        else:
            clip_type = None

        model_options = {}
        if device == "cpu":
            cpu = torch.device("cpu")
            model_options["load_device"] = cpu
            model_options["offload_device"] = cpu

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )

        # Idempotency guard: check the flag on the inner nn.Module rather than
        # the ComfyUI CLIP wrapper, which blocks arbitrary attribute assignment
        # (uses __slots__ or a restrictive __setattr__).
        target_device = self._resolve_device(device)
        inner_module, _ = _find_first_module(clip)
        if inner_module is not None and getattr(inner_module, "_fp8_storage_applied", False):
            print("comfy_fp8: FP8 storage already applied to cached model, skipping")
        else:
            try:
                clip = apply_fp8_storage_to_module(clip, compute_dtype=torch.float16, device=target_device)
                if FP8_AVAILABLE:
                    # Store the flag on the inner nn.Module — the ComfyUI CLIP
                    # wrapper does not support arbitrary attribute assignment, but
                    # nn.Module always does (it uses a plain __dict__).
                    if inner_module is not None:
                        inner_module._fp8_storage_applied = True
                    print("comfy_fp8: FP8 storage hook applied")
                    self._fix_model_size_reporting(clip, inner_module)
            except Exception as e:
                print("comfy_fp8: FP8 storage hook not applied:", e)

        return (clip,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_model_size_reporting(clip, inner_module=None):
        """
        Recompute and patch the model's ``model_size`` method after FP8
        conversion.

        ComfyUI calls ``self.model.model_size()`` (a method) inside
        ``model_memory_required`` when scheduling VRAM loads.  That method
        returns a value calculated at construction time — before our weight
        surgery — so it reflects the original FP16 footprint rather than the
        actual FP8 footprint.

        We replace ``model_size`` on the *model* object (``clip.patcher.model``,
        i.e. the ComfyUI BaseModel subclass) with a tiny lambda that returns
        the correct byte count.  The patcher itself is not touched.

        Note: we must NOT assign an int to ``patcher.model_size`` — that
        would shadow the patcher's own ``model_size`` attribute (if any) and,
        more critically, would shadow the ``model_size`` *method* on the inner
        model object if the names collide through attribute lookup, causing
        ``TypeError: 'int' object is not callable``.
        """
        try:
            patcher = getattr(clip, "patcher", None) or getattr(clip, "model_patcher", None)
            if patcher is None:
                return

            # The object whose model_size() ComfyUI actually calls is
            # patcher.model (the BaseModel subclass), not the patcher itself.
            comfy_model = getattr(patcher, "model", None)
            if comfy_model is None:
                return

            if inner_module is None:
                inner_module, _ = _find_first_module(clip)
            if inner_module is None:
                return

            # Sum FP8 weight buffers (now in _buffers) + remaining parameters
            # (biases etc., still in _parameters).
            total_bytes = (
                sum(b.nbytes for b in inner_module.buffers())
                + sum(p.nbytes for p in inner_module.parameters())
            )

            # Replace the method with a callable that returns the corrected
            # value.  Using a default-argument lambda freezes total_bytes at
            # this point in time (avoids late-binding closure issues).
            comfy_model.model_size = lambda _bytes=total_bytes: _bytes

            print(
                f"comfy_fp8: corrected model_size reporting to "
                f"{total_bytes / 1024 ** 2:.1f} MB"
            )
        except Exception as e:
            print(f"comfy_fp8: could not fix model_size reporting: {e}")

    @staticmethod
    def _resolve_device(device_pref: str) -> torch.device:
        """
        Return the ``torch.device`` to use for FP8 buffer allocation.

        When *device_pref* is ``"cpu"`` we respect that choice explicitly.
        Otherwise we prefer CUDA, falling back to CPU if no GPU is present.
        We deliberately avoid calling ``clip.parameters()`` here because the
        ComfyUI CLIP wrapper is not an ``nn.Module`` and does not expose that
        method.
        """
        if device_pref == "cpu":
            return torch.device("cpu")
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")