"""Utility functions for ComfyUI-PrimitiveAnything nodes."""

import os
import sys
import torch
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import folder_paths


# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}


def get_primitive_anything_path() -> Path:
    """Get the path to the PrimitiveAnything source code (vendored)."""
    return Path(__file__).parent.parent


def get_primitive_anything_models_path() -> Path:
    """Get the path to PrimitiveAnything models directory."""
    models_dir = Path(folder_paths.models_dir) / "primitive_anything"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def setup_primitive_anything_imports():
    """Add PrimitiveAnything to Python path for imports."""
    pa_path = get_primitive_anything_path()
    if str(pa_path) not in sys.path:
        sys.path.insert(0, str(pa_path))


def get_device() -> torch.device:
    """Get the appropriate torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_mesh_for_pa(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Normalize mesh for PrimitiveAnything processing.
    Scales to fit in [-1.6, 1.6] range (matching demo.py).

    Args:
        mesh: Input trimesh

    Returns:
        Normalized trimesh (copy)
    """
    mesh = mesh.copy()

    vertices = mesh.vertices
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max() * 1.6
    mesh.vertices = vertices

    return mesh


def tensors_to_trimesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    metadata: Optional[Dict[str, Any]] = None
) -> trimesh.Trimesh:
    """Convert vertex and face tensors to trimesh.Trimesh."""
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    if metadata:
        mesh.metadata.update(metadata)

    return mesh


# Primitive type mappings
CODE_SHAPE = {
    0: 'SM_GR_BS_CubeBevel_001.ply',
    1: 'SM_GR_BS_SphereSharp_001.ply',
    2: 'SM_GR_BS_CylinderSharp_001.ply',
}

SHAPENAME_MAP = {
    'SM_GR_BS_CubeBevel_001.ply': 1101002001034001,
    'SM_GR_BS_SphereSharp_001.ply': 1101002001034010,
    'SM_GR_BS_CylinderSharp_001.ply': 1101002001034002,
}
