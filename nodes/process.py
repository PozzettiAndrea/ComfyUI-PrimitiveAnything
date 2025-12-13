"""PrimitiveAnythingProcess node for running inference."""

import time
import json
import torch
import trimesh
import numpy as np
import seaborn as sns
from scipy.spatial.transform import Rotation
from typing import Any, Dict

from .utils import CODE_SHAPE, SHAPENAME_MAP, get_device


class PrimitiveAnythingProcess:
    """
    Run PrimitiveAnything inference to generate primitive assembly.

    This node takes the model and preprocessed point cloud data and
    generates a mesh composed of primitive shapes (cube, sphere, cylinder).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PA_MODEL", {
                    "tooltip": "PrimitiveAnything model from loader node"
                }),
                "preprocessed": ("PA_DATA", {
                    "tooltip": "Preprocessed data from PrimitiveAnything Preprocess node"
                }),
            },
            "optional": {
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
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "primitives_json")
    OUTPUT_TOOLTIPS = (
        "Assembled mesh from primitives",
        "JSON string with primitive parameters"
    )
    FUNCTION = "process"
    CATEGORY = "PrimitiveAnything"
    DESCRIPTION = "Generate primitive assembly from preprocessed mesh data."

    def process(
        self,
        model: Dict[str, Any],
        preprocessed: Dict[str, Any],
        temperature: float = 0.0,
        postprocess: str = "recon_loss",
        seed: int = 0,
    ):
        """Run PrimitiveAnything inference."""
        print(f"[PrimitiveAnything] Running inference (temp={temperature}, postprocess={postprocess})")

        # Set seed
        if seed > 0:
            from accelerate.utils import set_seed
            set_seed(seed)

        transformer = model["transformer"]
        accelerator = model["accelerator"]
        device = model["device"]
        mesh_bs = model["mesh_bs"]

        # Get preprocessed data
        pc_normal = preprocessed["pc_normal"]
        mesh_info = preprocessed["mesh_info"]

        # Convert to tensor
        input_pc = torch.tensor(pc_normal, dtype=torch.float16, device=device)[None]

        # Run inference
        print(f"[PrimitiveAnything] Running generation...")
        start_time = time.time()

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

        inference_time = time.time() - start_time
        print(f"[PrimitiveAnything] Inference completed in {inference_time:.2f}s")

        # Count primitives
        type_codes = recon_primitives['type_code'].squeeze().cpu().numpy()
        num_primitives = np.sum(type_codes != -1)
        print(f"[PrimitiveAnything] Generated {num_primitives} primitives")

        # Build output mesh and JSON
        output_mesh, primitives_json = self._build_output(
            recon_primitives, mesh_bs, mesh_info, inference_time
        )

        return (output_mesh, json.dumps(primitives_json, indent=2))

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

    def _build_output(
        self,
        primitives: Dict[str, torch.Tensor],
        mesh_bs: Dict[str, trimesh.Trimesh],
        mesh_info: Dict[str, Any],
        inference_time: float
    ):
        """Build output mesh and JSON from primitives."""
        out_json = {
            'operation': 0,
            'type': 1,
            'scene_id': None,
            'group': [],
            'metadata': {
                'inference_time': inference_time,
                'input_info': mesh_info,
            }
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

            # Add to JSON
            primitive_block = {
                'type_id': SHAPENAME_MAP[bs_name],
                'data': {
                    'location': translation.tolist(),
                    'rotation': self._euler_to_quat(rotation).tolist(),
                    'scale': scale.tolist(),
                    'color': ['808080']
                }
            }
            out_json['group'].append(primitive_block)

            # Build mesh if basic shapes available
            if bs_name in mesh_bs:
                trans_matrix = self._srt_to_matrix(
                    scale, self._euler_to_quat(rotation), translation
                )
                bs = mesh_bs[bs_name].copy().apply_transform(trans_matrix)

                # Apply color
                new_vertex_colors = np.repeat(
                    color_map[idx:idx+1],
                    bs.visual.vertex_colors.shape[0],
                    axis=0
                )
                bs.visual.vertex_colors[:, :3] = new_vertex_colors

                # Coordinate swap (Y <-> Z with sign flip)
                vertices = bs.vertices.copy()
                vertices[:, 1] = bs.vertices[:, 2]
                vertices[:, 2] = -bs.vertices[:, 1]
                bs.vertices = vertices

                model_scene.add_geometry(bs)

        # Convert scene to single mesh
        if len(model_scene.geometry) > 0:
            output_mesh = model_scene.dump(concatenate=True)
        else:
            # Return empty mesh if no primitives
            output_mesh = trimesh.Trimesh()

        # Add metadata
        output_mesh.metadata.update({
            'source': 'primitive_anything',
            'num_primitives': num_valid,
            'inference_time': inference_time,
            'input_info': mesh_info,
        })

        return output_mesh, out_json


NODE_CLASS_MAPPINGS = {
    "PrimitiveAnythingProcess": PrimitiveAnythingProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimitiveAnythingProcess": "PrimitiveAnything Process",
}
