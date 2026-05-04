"""3D U-Net model definition via MONAI.

Architecture:
- monai.networks.nets.UNet
- in_channels=4 (T1, T1ce, T2, FLAIR concatenated)
- out_channels=3 (WT, TC, ET — sigmoid per channel, NOT softmax)
- channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2), num_res_units=2
"""
