# MIT License
# Copyright (c) 2025 ComfyUI-PrimitiveAnything Contributors

"""
PrimitiveAnything PreStartup Script
- Auto-installs pytorch3d from miropsota wheels
- Copies example 3D assets to ComfyUI input folder on startup.
"""
import os
import shutil
import subprocess
import sys


def ensure_pytorch3d():
    """Auto-install pytorch3d if missing."""
    try:
        import pytorch3d
        return
    except ImportError:
        pass

    try:
        import torch
    except ImportError:
        return

    torch_ver = torch.__version__.split('+')[0]
    cuda_ver = torch.version.cuda

    if cuda_ver:
        pkg = f"pytorch3d==0.7.9+pt{torch_ver}cu{cuda_ver.replace('.', '')}"
    else:
        pkg = f"pytorch3d==0.7.9+pt{torch_ver}cpu"

    print(f"[PrimitiveAnything] Auto-installing {pkg}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "--extra-index-url", "https://miropsota.github.io/torch_packages_builder",
            pkg
        ])
        print("[PrimitiveAnything] pytorch3d installed")
    except Exception as e:
        print(f"[PrimitiveAnything] Failed to install pytorch3d: {e}")


def copy_example_assets():
    """Copy all files and folders from assets/ directory to ComfyUI input/3d directory."""
    try:
        import folder_paths

        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))

        # Create input/3d subdirectory
        input_3d_folder = os.path.join(input_folder, "3d")
        os.makedirs(input_3d_folder, exist_ok=True)

        # Copy entire assets/ folder structure
        assets_folder = os.path.join(custom_node_dir, "assets")
        if not os.path.exists(assets_folder):
            print(f"[PrimitiveAnything] Warning: assets folder not found at {assets_folder}")
            return

        copied_count = 0
        for root, dirs, files in os.walk(assets_folder):
            # Calculate relative path from assets folder
            rel_path = os.path.relpath(root, assets_folder)

            # Create corresponding subdirectory in destination
            if rel_path != '.':
                dest_dir = os.path.join(input_3d_folder, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
            else:
                dest_dir = input_3d_folder

            # Copy files
            for file in files:
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                if not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                    rel_dest = os.path.join(rel_path, file) if rel_path != '.' else file
                    print(f"[PrimitiveAnything] Copied {rel_dest} to input/3d/")

        if copied_count > 0:
            print(f"[PrimitiveAnything] [OK] Copied {copied_count} asset(s) to {input_3d_folder}")
        else:
            print(f"[PrimitiveAnything] All assets already exist in {input_3d_folder}")

    except Exception as e:
        print(f"[PrimitiveAnything] Error copying assets: {e}")


# Run on import
ensure_pytorch3d()
copy_example_assets()
