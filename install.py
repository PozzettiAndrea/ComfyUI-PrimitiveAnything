"""Auto-install pytorch3d from miropsota wheels for ComfyUI-PrimitiveAnything."""

import subprocess
import sys

def main():
    # Check if already installed
    try:
        import pytorch3d
        print(f"[ComfyUI-PrimitiveAnything] pytorch3d {pytorch3d.__version__} already installed")
        return
    except ImportError:
        pass

    # Get torch info
    try:
        import torch
    except ImportError:
        print("[ComfyUI-PrimitiveAnything] PyTorch not found, skipping pytorch3d install")
        return

    torch_ver = torch.__version__.split('+')[0]
    cuda_ver = torch.version.cuda

    if cuda_ver:
        cuda_short = cuda_ver.replace('.', '')
        pkg = f"pytorch3d==0.7.9+pt{torch_ver}cu{cuda_short}"
    else:
        pkg = f"pytorch3d==0.7.9+pt{torch_ver}cpu"

    print(f"[ComfyUI-PrimitiveAnything] Installing {pkg}")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--extra-index-url", "https://miropsota.github.io/torch_packages_builder",
        pkg
    ])
    print("[ComfyUI-PrimitiveAnything] pytorch3d installed")

if __name__ == "__main__":
    main()
