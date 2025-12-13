"""
ComfyUI-PrimitiveAnything: ComfyUI nodes for PrimitiveAnything 3D primitive decomposition.

PrimitiveAnything decomposes 3D shapes into assemblies of basic primitives
(cube, sphere, cylinder) using an auto-regressive transformer.

Nodes:
- (Down)Load PrimitiveAnything Model: Download and load the transformer model
- PrimitiveAnything Preprocess: Prepare mesh (normalize, watertight, point cloud)
- PrimitiveAnything Process: Run inference to generate primitive assembly

Integrates with GeometryPack's TRIMESH type for seamless mesh pipeline.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
