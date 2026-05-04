"""Loss function construction.

Design notes:
- sigmoid=True: the three tumour regions (TC, WT, ET) are nested/overlapping,
  so each channel is an independent binary decision. Softmax would enforce
  mutual exclusivity and is wrong here.
- to_onehot_y=False: the label tensor is already multi-channel binary after
  ConvertToMultiChannelBasedOnBratsClassesd; one-hot conversion must be skipped
  or it would misinterpret the existing channels as class indices.
"""

from monai.losses import DiceCELoss


def build_loss() -> DiceCELoss:
    return DiceCELoss(
        sigmoid=True,
        to_onehot_y=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
