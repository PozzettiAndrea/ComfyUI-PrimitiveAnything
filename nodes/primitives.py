"""Merged PrimitiveAnything node with automatic denormalization."""

import os
import sys
import time
import json
import torch
import trimesh
import numpy as np
import seaborn as sns
import skimage.measure
from scipy.spatial.transform import Rotation
from typing import Any, Dict, Optional

from .utils import (
    CODE_SHAPE,
    SHAPENAME_MAP,
    get_device,
    normalize_mesh_for_pa,
    setup_primitive_anything_imports,
)


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


class PrimitiveAnything:
    """
    All-in-one PrimitiveAnything node with automatic denormalization.

    This node combines preprocessing, inference, and denormalization into a
    single workflow. The output mesh is automatically scaled and positioned
    to match the original input mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH", {
                    "tooltip": "Input mesh from GeometryPack"
                }),
                "model": ("PA_MODEL", {
                    "tooltip": "PrimitiveAnything model from loader node"
                }),
            },
            "optional": {
                # Preprocessing parameters
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
                # Processing parameters
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature (0.0 = greedy/deterministic)"
                }),
                "postprocess": (["none", "recon_loss"], {
                    "default": "recon_loss",
                    "tooltip": "Postprocessing: recon_loss uses reconstruction loss for better results"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed for sampling"
                }),
                # Denormalization control
                "denormalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Denormalize output to original scale/position (recommended)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "primitives_json")
    OUTPUT_TOOLTIPS = (
        "Assembled mesh from primitives (in original scale/position if denormalize=True)",
        "JSON string with primitive parameters (both normalized and denormalized)"
    )
    FUNCTION = "process"
    CATEGORY = "PrimitiveAnything"
    DESCRIPTION = "All-in-one PrimitiveAnything: preprocess, inference, and automatic denormalization to original scale."

    def process(
        self,
        mesh: trimesh.Trimesh,
        model: Dict[str, Any],
        marching_cubes: bool = True,
        dilated_offset: float = 0.015,
        sample_points: int = 10000,
        temperature: float = 0.0,
        postprocess: str = "recon_loss",
        seed: int = 0,
        denormalize: bool = True,
    ):
        """Run full PrimitiveAnything pipeline with optional denormalization."""
        print(f"[PrimitiveAnything] Starting pipeline (denormalize={denormalize})")

        # STEP 1: Extract normalization parameters BEFORE preprocessing
        normalization_params = self._extract_normalization_params(mesh) if denormalize else None
        if normalization_params:
            print(f"[PrimitiveAnything] Original mesh - center: {normalization_params['center']}, extent: {normalization_params['max_extent']:.4f}")

        # STEP 2: Preprocess mesh
        pc_normal, processed_mesh, mesh_info = self._preprocess(
            mesh, marching_cubes, dilated_offset, sample_points
        )

        # STEP 3: Run inference
        if seed > 0:
            from accelerate.utils import set_seed
            set_seed(seed)

        start_time = time.time()
        recon_primitives, mask = self._run_inference(
            pc_normal, model, temperature, postprocess
        )
        inference_time = time.time() - start_time

        # Count primitives
        type_codes = recon_primitives['type_code'].squeeze().cpu().numpy()
        num_primitives = np.sum(type_codes != -1)
        print(f"[PrimitiveAnything] Generated {num_primitives} primitives")

        # STEP 4: Build output with optional denormalization
        output_mesh, primitives_json = self._build_output(
            recon_primitives,
            model["mesh_bs"],
            mesh_info,
            inference_time,
            normalization_params  # None if denormalize=False
        )

        return (output_mesh, json.dumps(primitives_json, indent=2))

    def _extract_normalization_params(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Extract parameters needed to denormalize back to original scale.

        This must be called BEFORE any preprocessing/normalization.
        """
        if len(mesh.vertices) == 0:
            # Empty mesh - return identity transform
            return {
                "center": np.zeros(3, dtype=np.float64),
                "max_extent": 1.0,
                "scale_factor": 1.6,
            }

        vertices = mesh.vertices
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        center = (bounds[0] + bounds[1]) / 2
        max_extent = (bounds[1] - bounds[0]).max()

        return {
            "center": center.astype(np.float64),
            "max_extent": float(max_extent),
            "scale_factor": 1.6,
        }

    def _preprocess(
        self,
        mesh: trimesh.Trimesh,
        marching_cubes: bool,
        dilated_offset: float,
        sample_points: int,
    ):
        """Preprocessing logic from PrimitiveAnythingPreprocess."""
        print(f"[PrimitiveAnything] Preprocessing: mc={marching_cubes}, dilate={dilated_offset}")

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

        print(f"[PrimitiveAnything] Preprocessing complete: {len(pc_normal)} points")

        return pc_normal.astype(np.float16), normalized_mesh, mesh_info

    def _run_inference(
        self,
        pc_normal: np.ndarray,
        model: Dict[str, Any],
        temperature: float,
        postprocess: str,
    ):
        """Inference logic from PrimitiveAnythingProcess."""
        transformer = model["transformer"]
        accelerator = model["accelerator"]
        device = model["device"]

        # Convert to tensor
        input_pc = torch.tensor(pc_normal, dtype=torch.float16, device=device)[None]

        # Run inference
        print(f"[PrimitiveAnything] Running generation (temp={temperature}, postprocess={postprocess})...")

        with torch.no_grad():
            with accelerator.autocast():
                if postprocess == "recon_loss":
                    recon_primitives, mask = transformer.generate_w_recon_loss(
                        pc=input_pc,
                        temperature=temperature,
                        single_directional=True
                    )
                else:
                    recon_primitives, mask = transformer.generate(
                        pc=input_pc,
                        temperature=temperature
                    )

        return recon_primitives, mask

    def _build_output(
        self,
        primitives: Dict[str, torch.Tensor],
        mesh_bs: Dict[str, trimesh.Trimesh],
        mesh_info: Dict[str, Any],
        inference_time: float,
        normalization_params: Optional[Dict[str, Any]],
    ):
        """Build output mesh and JSON with optional denormalization."""
        out_json = {
            'operation': 0,
            'type': 1,
            'scene_id': None,
            'group': [],
            'metadata': {
                'inference_time': inference_time,
                'input_info': mesh_info,
                'denormalized': normalization_params is not None,
            }
        }

        if normalization_params:
            out_json['metadata']['normalization_params'] = {
                'center': normalization_params['center'].tolist(),
                'max_extent': normalization_params['max_extent'],
                'scale_factor': normalization_params['scale_factor'],
            }

        model_scene = trimesh.Scene()

        # Get primitive data
        scales = primitives['scale'].squeeze().cpu().numpy()
        rotations = primitives['rotation'].squeeze().cpu().numpy()
        translations = primitives['translation'].squeeze().cpu().numpy()
        type_codes = primitives['type_code'].squeeze().cpu().numpy()

        num_valid = np.sum(type_codes != -1)
        color_map = sns.color_palette("hls", max(num_valid, 1))
        color_map = (np.array(color_map) * 255).astype("uint8")

        for idx, (scale, rotation, translation, type_code) in enumerate(
            zip(scales, rotations, translations, type_codes)
        ):
            if type_code == -1:
                break

            bs_name = CODE_SHAPE[int(type_code)]

            # Build JSON with both normalized and denormalized parameters
            primitive_block = {
                'type_id': SHAPENAME_MAP[bs_name],
                'data': {
                    'location': translation.tolist(),  # Normalized
                    'rotation': self._euler_to_quat(rotation).tolist(),
                    'scale': scale.tolist(),  # Normalized
                    'color': ['808080']
                }
            }

            # Add denormalized parameters to JSON if enabled
            if normalization_params is not None:
                denorm_trans = self._denormalize_translation(translation, normalization_params)
                denorm_scale = self._denormalize_scale(scale, normalization_params)
                primitive_block['data']['denormalized'] = {
                    'location': denorm_trans.tolist(),
                    'scale': denorm_scale.tolist(),
                }

            out_json['group'].append(primitive_block)

            # Build mesh if basic shapes available
            if bs_name in mesh_bs:
                # Step 1: Apply SRT transformation (normalized space)
                trans_matrix = self._srt_to_matrix(
                    scale, self._euler_to_quat(rotation), translation
                )
                bs = mesh_bs[bs_name].copy().apply_transform(trans_matrix)

                # Step 2: Apply color
                new_vertex_colors = np.repeat(
                    color_map[idx:idx+1],
                    bs.visual.vertex_colors.shape[0],
                    axis=0
                )
                bs.visual.vertex_colors[:, :3] = new_vertex_colors

                # Step 3: Coordinate swap (Y ↔ Z with sign flip)
                # CRITICAL: This must happen BEFORE denormalization
                vertices = bs.vertices.copy()
                vertices[:, 1] = bs.vertices[:, 2]
                vertices[:, 2] = -bs.vertices[:, 1]
                bs.vertices = vertices

                # Step 4: Denormalize vertices (if enabled)
                # This transforms from normalized space [-1.6, 1.6] back to original scale
                if normalization_params is not None:
                    bs.vertices = self._denormalize_vertices(bs.vertices, normalization_params)

                model_scene.add_geometry(bs)

        # Convert scene to single mesh
        if len(model_scene.geometry) > 0:
            output_mesh = model_scene.dump(concatenate=True)
        else:
            # Return empty mesh if no primitives
            output_mesh = trimesh.Trimesh()

        # Add metadata to output mesh
        output_mesh.metadata.update({
            'source': 'primitive_anything',
            'num_primitives': num_valid,
            'inference_time': inference_time,
            'input_info': mesh_info,
            'denormalized': normalization_params is not None,
        })

        if normalization_params:
            output_mesh.metadata.update({
                'original_center': normalization_params['center'].tolist(),
                'original_max_extent': normalization_params['max_extent'],
            })

        print(f"[PrimitiveAnything] Output built: {len(output_mesh.vertices)} vertices, denormalized={normalization_params is not None}")

        return output_mesh, out_json

    def _denormalize_vertices(
        self,
        vertices: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Denormalize vertices from normalized space [-1.6, 1.6] to original scale.

        Reverse of: normalized = (vertices - center) / max_extent * 1.6
        Formula: original = (normalized / 1.6) * max_extent + center
        """
        return (vertices / params["scale_factor"]) * params["max_extent"] + params["center"]

    def _denormalize_translation(
        self,
        translation: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Denormalize translation vector."""
        return (translation / params["scale_factor"]) * params["max_extent"] + params["center"]

    def _denormalize_scale(
        self,
        scale: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Denormalize scale factor.

        In normalized space, scale is relative to 1.6 range.
        In original space, scale should be relative to max_extent.
        """
        return scale * params["max_extent"] / params["scale_factor"]

    # Helper methods from PrimitiveAnythingProcess

    def _euler_to_quat(self, euler):
        """Convert Euler angles to quaternion."""
        return Rotation.from_euler('XYZ', euler, degrees=True).as_quat()

    def _srt_to_matrix(self, scale, quat, translation):
        """Create transformation matrix from scale, rotation (quat), translation."""
        rotation_matrix = Rotation.from_quat(quat).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix * scale
        transform_matrix[:3, 3] = translation
        return transform_matrix

    # Helper methods from PrimitiveAnythingPreprocess

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
    "PrimitiveAnything": PrimitiveAnything,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimitiveAnything": "PrimitiveAnything",
}
