"""Streamlit demo: upload 4 NIfTI modalities, get a segmentation overlay.

Usage:
    streamlit run app/streamlit_app.py

UI flow:
- File uploaders for T1, T1ce, T2, FLAIR NIfTI files
- Slice slider to pick the axial slice to display
- Runs predict.py inference on upload
- Renders overlay: background MRI + color-coded WT/TC/ET masks
"""
