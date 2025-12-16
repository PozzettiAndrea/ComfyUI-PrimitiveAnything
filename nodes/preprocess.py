"""PrimitiveAnythingPreprocess node for preparing meshes."""

import os
import sys
import time
import torch
import numpy as np
import trimesh
import skimage.measure
from typing import Any, Dict

from .utils import normalize_mesh_for_pa, setup_primitive_anything_imports


def _setup_opengl_platform():
    """Auto-detect and configure OpenGL platform for headless rendering."""
    # Already configured - respect user's choice
    if 'PYOPENGL_PLATFORM' in os.environ:
        return

    # Not Linux - use default (works on Windows/Mac with display)
    if sys.platform != 'linux':
        return

    # Linux with display - use default Pyglet
    if os.environ.get('DISPLAY'):
        return

    # Headless Linux - try EGL first (GPU), then OSMesa (software)
    for platform in ['egl', 'osmesa']:
        os.environ['PYOPENGL_PLATFORM'] = platform
        try:
            import OpenGL.GL  # noqa - test if platform works
            print(f"[PrimitiveAnything] Using OpenGL platform: {platform}")
            return
        except Exception:
            pass

    # Clear if nothing worked - will fall back to trimesh sampling
    os.environ.pop('PYOPENGL_PLATFORM', None)
    print("[PrimitiveAnything] Warning: No OpenGL platform available, will use trimesh fallback")


# Configure OpenGL platform before any rendering imports
_setup_opengl_platform()


class PrimitiveAnythingPreprocess:
    """
    Prepare mesh for PrimitiveAnything processing.

    This node normalizes the mesh, optionally makes it watertight via
    marching cubes, and generates a surface point cloud with normals.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH", {
                    "tooltip": "Input mesh from GeometryPack"
                }),
            },
            "optional": {
                "marching_cubes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Convert to watertight mesh using marching cubes (recommended)"
                }),
                "dilated_offset": ("FLOAT", {
                    "default": 0.015,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Dilate mesh along normals (helps with thin features)"
                }),
                "sample_points": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 100000,
                    "step": 1000,
                    "tooltip": "Number of surface points to sample"
                }),
            }
        }

    RETURN_TYPES = ("PA_DATA",)
    RETURN_NAMES = ("preprocessed",)
    OUTPUT_TOOLTIPS = ("Preprocessed point cloud data for PrimitiveAnything",)
    FUNCTION = "preprocess"
    CATEGORY = "PrimitiveAnything"
    DESCRIPTION = "Prepare mesh for PrimitiveAnything: normalize, watertight conversion, point cloud sampling."

    def preprocess(
        self,
        mesh: trimesh.Trimesh,
        marching_cubes: bool = True,
        dilated_offset: float = 0.015,
        sample_points: int = 10000,
    ):
        """Preprocess mesh for PrimitiveAnything inference."""
        print(f"[PrimitiveAnything] Preprocessing mesh: mc={marching_cubes}, dilate={dilated_offset}")

        # Setup imports
        setup_primitive_anything_imports()

        # Normalize mesh to [-1.6, 1.6] range
        normalized_mesh = normalize_mesh_for_pa(mesh)

        # Convert to watertight if requested
        if marching_cubes:
            print("[PrimitiveAnything] Running marching cubes...")
            normalized_mesh = self._export_to_watertight(normalized_mesh)

        # Apply dilation
        if dilated_offset > 0:
            print(f"[PrimitiveAnything] Applying dilation: {dilated_offset}")
            normalized_mesh.vertices = normalized_mesh.vertices + normalized_mesh.vertex_normals * dilated_offset

        # Clean up mesh
        normalized_mesh.merge_vertices()
        normalized_mesh.update_faces(normalized_mesh.unique_faces())
        normalized_mesh.fix_normals()

        # Sample surface points with normals
        print(f"[PrimitiveAnything] Sampling {sample_points} surface points...")
        pc_normal = self._sample_surface_points(normalized_mesh, sample_points)

        # Re-normalize if dilated
        if dilated_offset > 0:
            vertices = normalized_mesh.vertices
            bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            center = (bounds[0] + bounds[1]) / 2
            max_extent = (bounds[1] - bounds[0]).max()

            normalized_mesh.vertices = (vertices - center) / max_extent * 1.6
            pc_normal[:, :3] = (pc_normal[:, :3] - center) / max_extent * 1.6

        # Verify normals are unit vectors
        normals = pc_normal[:, 3:]
        norms = np.linalg.norm(normals, axis=-1)
        if not (norms > 0.99).all():
            print("[PrimitiveAnything] Warning: Re-normalizing normals")
            normals = normals / (norms[:, None] + 1e-8)
            pc_normal[:, 3:] = normals

        mesh_info = {
            "original_vertices": len(mesh.vertices),
            "original_faces": len(mesh.faces),
            "processed_vertices": len(normalized_mesh.vertices),
            "processed_faces": len(normalized_mesh.faces),
            "sample_points": sample_points,
            "marching_cubes": marching_cubes,
            "dilated_offset": dilated_offset,
        }

        preprocessed_data = {
            "pc_normal": pc_normal.astype(np.float16),
            "processed_mesh": normalized_mesh,
            "mesh_info": mesh_info,
        }

        print(f"[PrimitiveAnything] Preprocessing complete: {len(pc_normal)} points")

        return (preprocessed_data,)

    def _normalize_vertices(self, vertices, scale=0.9):
        """Normalize vertices to fit in bounding box."""
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale_factor = 2.0 * scale / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale_factor
        return vertices, center, scale_factor

    def _export_to_watertight(self, mesh: trimesh.Trimesh, octree_depth: int = 7) -> trimesh.Trimesh:
        """Convert mesh to watertight using marching cubes."""
        import mesh2sdf.core

        size = 2 ** octree_depth
        level = 2 / size

        scaled_vertices, to_orig_center, to_orig_scale = self._normalize_vertices(mesh.vertices)

        sdf = mesh2sdf.core.compute(scaled_vertices, mesh.faces, size=size)
        vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)

        # Scale back
        vertices = vertices / size * 2 - 1
        vertices = vertices / to_orig_scale + to_orig_center

        return trimesh.Trimesh(vertices, faces, normals=normals)

    def _sample_surface_points(self, mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
        """Sample surface points with normals from mesh."""
        try:
            from mesh_to_sdf import get_surface_point_cloud

            surface_pc = get_surface_point_cloud(
                mesh,
                surface_point_method='scan',
                bounding_radius=None,
                scan_count=100,
                scan_resolution=400,
                sample_point_count=10000000,
                calculate_normals=True
            )

            rng = np.random.default_rng()
            indices = rng.choice(surface_pc.points.shape[0], num_points, replace=True)
            points = surface_pc.points[indices]
            normals = surface_pc.normals[indices]

            return np.concatenate([points, normals], axis=-1).astype(np.float32)

        except ImportError:
            # Fallback to trimesh sampling
            print("[PrimitiveAnything] mesh_to_sdf not available, using trimesh sampling")
            points, face_indices = mesh.sample(num_points, return_index=True)
            normals = mesh.face_normals[face_indices]
            return np.concatenate([points, normals], axis=-1).astype(np.float32)


NODE_CLASS_MAPPINGS = {
    "PrimitiveAnythingPreprocess": PrimitiveAnythingPreprocess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimitiveAnythingPreprocess": "PrimitiveAnything Preprocess",
}
