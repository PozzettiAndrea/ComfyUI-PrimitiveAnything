"""DownloadAndLoadPrimitiveAnythingModel node."""

import os
import glob
import yaml
import torch
import trimesh
import numpy as np
from pathlib import Path
from typing import Any, Dict

from .utils import (
    _MODEL_CACHE,
    get_primitive_anything_path,
    get_primitive_anything_models_path,
    setup_primitive_anything_imports,
    get_device,
    CODE_SHAPE,
)


class DownloadAndLoadPrimitiveAnythingModel:
    """
    Download (if needed) and load the PrimitiveAnything model.

    This node downloads the transformer checkpoint and Michelangelo encoder
    from HuggingFace, then loads the model for inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["PrimitiveAnything"], {
                    "default": "PrimitiveAnything",
                    "tooltip": "PrimitiveAnything model for primitive decomposition"
                }),
            },
        }

    RETURN_TYPES = ("PA_MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("PrimitiveAnything model for generating primitive assemblies",)
    FUNCTION = "load_model"
    CATEGORY = "PrimitiveAnything"
    DESCRIPTION = "Download and load PrimitiveAnything model for 3D primitive decomposition."

    def load_model(self, model: str):
        """Load the PrimitiveAnything model."""
        print(f"[PrimitiveAnything] Loading model: {model}")

        device = get_device()

        # Check CUDA
        if device.type == "cpu":
            print("[PrimitiveAnything] WARNING: CUDA not available, running on CPU will be slow!")

        # Cache key
        cache_key = f"{model}"

        if cache_key in _MODEL_CACHE:
            print(f"[PrimitiveAnything] Using cached model")
            return (_MODEL_CACHE[cache_key],)

        # Download checkpoints if needed
        ckpt_path = self._get_or_download_checkpoint()

        # Setup imports
        setup_primitive_anything_imports()

        # Load config
        pa_path = get_primitive_anything_path()
        config_path = pa_path / "configs" / "infer.yml"

        with open(config_path, 'r') as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)

        # Create model
        from primitive_anything.primitive_transformer import PrimitiveTransformerDiscrete

        print(f"[PrimitiveAnything] Creating model...")
        model_cfg = config['model'].copy()
        model_cfg.pop('name', None)
        transformer = PrimitiveTransformerDiscrete(**model_cfg)

        # Load checkpoint
        print(f"[PrimitiveAnything] Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        transformer.load_state_dict(checkpoint)

        # Prepare with accelerator for fp16
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision='fp16')
        transformer = accelerator.prepare(transformer)
        transformer.eval()

        # Move auxiliary tensors to GPU
        if hasattr(transformer, 'bs_pc'):
            transformer.bs_pc = transformer.bs_pc.to(device)
        if hasattr(transformer, 'rotation_matrix_align_coord'):
            transformer.rotation_matrix_align_coord = transformer.rotation_matrix_align_coord.to(device)

        # Load basic shapes for rendering
        bs_dir = pa_path / "data" / "basic_shapes_norm"
        mesh_bs = {}
        if bs_dir.exists():
            for bs_path in glob.glob(str(bs_dir / "*.ply")):
                bs_name = os.path.basename(bs_path)
                bs = trimesh.load(bs_path)
                if hasattr(bs.visual, 'uv') and bs.visual.uv is not None:
                    bs.visual.uv = np.clip(bs.visual.uv, 0, 1)
                bs.visual = bs.visual.to_color()
                mesh_bs[bs_name] = bs
            print(f"[PrimitiveAnything] Loaded {len(mesh_bs)} basic shapes")
        else:
            print(f"[PrimitiveAnything] WARNING: Basic shapes not found at {bs_dir}")

        # Create model wrapper
        model_wrapper = {
            "transformer": transformer,
            "accelerator": accelerator,
            "config": config,
            "device": device,
            "mesh_bs": mesh_bs,
        }

        _MODEL_CACHE[cache_key] = model_wrapper
        print(f"[PrimitiveAnything] Model loaded successfully")

        return (model_wrapper,)

    @classmethod
    def _get_or_download_checkpoint(cls) -> Path:
        """Get checkpoint path, downloading if necessary."""
        models_dir = get_primitive_anything_models_path()
        ckpt_path = models_dir / "mesh-transformer.ckpt.60.pt"

        # Also check in PrimitiveAnything/ckpt
        pa_path = get_primitive_anything_path()
        pa_ckpt = pa_path / "ckpt" / "mesh-transformer.ckpt.60.pt"

        if pa_ckpt.exists():
            print(f"[PrimitiveAnything] Found checkpoint in source: {pa_ckpt}")
            return pa_ckpt

        if ckpt_path.exists():
            print(f"[PrimitiveAnything] Found checkpoint: {ckpt_path}")
            return ckpt_path

        # Download from HuggingFace
        print(f"[PrimitiveAnything] Downloading checkpoint from HuggingFace...")
        cls._download_checkpoint(models_dir)

        if not ckpt_path.exists():
            raise RuntimeError(f"Download completed but checkpoint not found: {ckpt_path}")

        return ckpt_path

    @classmethod
    def _download_checkpoint(cls, target_dir: Path):
        """Download checkpoint from HuggingFace."""
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download

            repo_id = "hyz317/PrimitiveAnything"

            print(f"[PrimitiveAnything] Downloading from {repo_id}...")

            # Download main checkpoint
            hf_hub_download(
                repo_id=repo_id,
                filename="mesh-transformer.ckpt.60.pt",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

            print(f"[PrimitiveAnything] Download complete")

        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading checkpoints. "
                "Please install it: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}") from e


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadPrimitiveAnythingModel": DownloadAndLoadPrimitiveAnythingModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadPrimitiveAnythingModel": "(Down)Load PrimitiveAnything Model",
}
