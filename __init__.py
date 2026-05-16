from .load_clip_fp8 import LoadCLIPFP8

NODE_CLASS_MAPPINGS = {
    "LoadCLIPFP8": LoadCLIPFP8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCLIPFP8": "Load CLIP FP8",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
__version__ = "0.1.0"
__author__ = "Dryw Filtiarn"
