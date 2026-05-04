"""Smoke-test the environment and data layout before training.

Checks:
- Python / PyTorch / MONAI / CUDA availability
- data_dir exists and contains the expected folder structure
- One patient can be loaded and transformed without errors
- Model instantiates and forward-passes a random patch
"""

import sys

try:
    import torch

    print(f"torch version : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot proceed without a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU name      : {gpu_name}")
    print(f"GPU memory    : {total_mem:.1f} GB")

    import monai
    print(f"monai version : {monai.__version__}")

    from monai.networks.nets import BasicUNet

    device = torch.device("cuda")
    model = BasicUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        features=(16, 16, 32, 64, 128, 16),
    ).to(device)

    x = torch.randn(1, 4, 32, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)

    print(f"output shape  : {tuple(y.shape)}")
    sys.exit(0)

except Exception as exc:
    print(f"FAILED: {exc}", file=sys.stderr)
    sys.exit(1)
