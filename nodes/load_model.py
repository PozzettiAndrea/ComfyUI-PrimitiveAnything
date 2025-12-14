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
        from primitive_anything.utils import safe_torch_load
        checkpoint = safe_torch_load(ckpt_path, map_location='cpu')
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

        # Load basic shapes for rendering (from ComfyUI models directory)
        models_dir = get_primitive_anything_models_path()
        bs_dir = models_dir / "basic_shapes_norm"
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
        """Get checkpoint path, downloading all dependencies if necessary."""
        # Use ComfyUI models directory for all checkpoints
        models_dir = get_primitive_anything_models_path()
        ckpt_path = models_dir / "mesh-transformer.ckpt.60.pt"

        # Download main checkpoint if needed
        if not ckpt_path.exists():
            print(f"[PrimitiveAnything] Downloading checkpoint from HuggingFace...")
            cls._download_main_checkpoint(models_dir)

        # Download Michelangelo encoder if needed
        encoder_path = models_dir / "shapevae-256.ckpt"
        if not encoder_path.exists():
            print(f"[PrimitiveAnything] Downloading Michelangelo encoder...")
            cls._download_michelangelo_encoder(models_dir)

        # Download basic shapes if needed
        shapes_dir = models_dir / "basic_shapes_norm"
        if not shapes_dir.exists() or not any(shapes_dir.glob("*.ply")):
            print(f"[PrimitiveAnything] Downloading basic shapes...")
            cls._download_basic_shapes(models_dir)

        if not ckpt_path.exists():
            raise RuntimeError(f"Download completed but checkpoint not found: {ckpt_path}")

        return ckpt_path

    @classmethod
    def _download_main_checkpoint(cls, target_dir: Path):
        """Download main transformer checkpoint from HuggingFace."""
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download

            repo_id = "hyz317/PrimitiveAnything"
            print(f"[PrimitiveAnything] Downloading from {repo_id}...")

            hf_hub_download(
                repo_id=repo_id,
                filename="mesh-transformer.ckpt.60.pt",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[PrimitiveAnything] Main checkpoint download complete")

        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading. "
                "Please install it: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Main checkpoint download failed: {e}") from e

    @classmethod
    def _download_michelangelo_encoder(cls, target_dir: Path):
        """Download Michelangelo ShapeVAE encoder from HuggingFace."""
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download

            repo_id = "Maikou/Michelangelo"
            print(f"[PrimitiveAnything] Downloading encoder from {repo_id}...")

            # Download to temp location then move to target
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename="checkpoints/aligned_shape_latents/shapevae-256.ckpt",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

            # Move from nested path to ckpt dir
            import shutil
            nested_path = target_dir / "checkpoints" / "aligned_shape_latents" / "shapevae-256.ckpt"
            if nested_path.exists():
                shutil.move(str(nested_path), str(target_dir / "shapevae-256.ckpt"))
                # Clean up nested dirs
                shutil.rmtree(str(target_dir / "checkpoints"), ignore_errors=True)

            print(f"[PrimitiveAnything] Michelangelo encoder download complete")

        except Exception as e:
            print(f"[PrimitiveAnything] WARNING: Michelangelo encoder download failed: {e}")
            print(f"[PrimitiveAnything] The model may still work without it.")

    @classmethod
    def _download_basic_shapes(cls, target_dir: Path):
        """Download basic primitive shapes from HuggingFace datasets."""
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download

            repo_id = "hyz317/PrimitiveAnything"
            files_to_download = [
                "basic_shapes_norm/SM_GR_BS_CubeBevel_001.ply",
                "basic_shapes_norm/SM_GR_BS_CylinderSharp_001.ply",
                "basic_shapes_norm/SM_GR_BS_SphereSharp_001.ply",
                "basic_shapes_norm/basic_shapes.json",
            ]

            print(f"[PrimitiveAnything] Downloading basic shapes from {repo_id} (dataset)...")

            for filename in files_to_download:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                )
                print(f"[PrimitiveAnything]   Downloaded: {filename}")

            print(f"[PrimitiveAnything] Basic shapes download complete")

        except Exception as e:
            raise RuntimeError(f"Basic shapes download failed: {e}") from e


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadPrimitiveAnythingModel": DownloadAndLoadPrimitiveAnythingModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadPrimitiveAnythingModel": "(Down)Load PrimitiveAnything Model",
}
