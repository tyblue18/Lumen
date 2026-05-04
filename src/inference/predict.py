"""Single-case inference with sliding-window strategy.

Responsibilities:
- Load a checkpoint and instantiate the model in eval mode
- Accept 4 NIfTI paths (T1, T1ce, T2, FLAIR)
- Apply validation transforms and run monai.inferers.SlidingWindowInferer
- Post-process logits → binary masks → label volume
- Write output segmentation as NIfTI
"""
