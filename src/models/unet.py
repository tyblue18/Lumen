"""3D U-Net model definition via MONAI.

Architecture:
- monai.networks.nets.UNet
- in_channels=4 (T1, T1ce, T2, FLAIR concatenated)
- out_channels=3 (TC, WT, ET — sigmoid applied by loss, NOT in the model)
- channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2), num_res_units=2
"""

from monai.networks.nets import UNet


def build_unet(config: dict | None = None) -> UNet:
    """Return a 3-D U-Net configured for BraTS segmentation.

    All hyper-parameters are fixed to the values in CLAUDE.md / default.yaml.
    *config* is accepted for future extension but currently unused.
    Sigmoid is intentionally absent from the model; DiceCELoss(sigmoid=True)
    applies it internally so the raw logits are also usable for inference
    thresholding without double-sigmoid.
    """
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
    )
