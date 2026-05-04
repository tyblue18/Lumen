"""Run inference on a single patient and write a segmentation NIfTI.

Usage:
    python scripts/predict.py \\
        --t1    path/to/t1.nii.gz \\
        --t1ce  path/to/t1ce.nii.gz \\
        --t2    path/to/t2.nii.gz \\
        --flair path/to/flair.nii.gz \\
        --ckpt  runs/default/best_model.pth \\
        --out   output_seg.nii.gz
"""
