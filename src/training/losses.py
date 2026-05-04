"""Loss function construction.

Uses monai.losses.DiceCELoss with sigmoid=True.
Regions are nested/overlapping (WT ⊇ TC ⊇ ET), so sigmoid per channel
is correct — do not use softmax.
"""
