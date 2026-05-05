# Lumen — BraTS 2020 Brain-Tumour Segmentation

3-D patch-based segmentation of glioma sub-regions using a MONAI UNet trained on
BraTS 2020 multi-modal MRI. Outputs three overlapping binary masks for whole
tumour (WT), tumour core (TC), and enhancing tumour (ET).

## Quick start

```bash
# Create and activate the venv
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Smoke-test (must exit 0 before training)
python scripts/verify_setup.py

# Train
python scripts/train.py --config configs/default.yaml --run-dir runs/exp1

# Evaluate a checkpoint
python scripts/evaluate.py --ckpt runs/exp1/best.pt

# Predict on one patient
python scripts/predict.py \
    --t1    data/raw/.../BraTS20_Training_001_t1.nii \
    --t1ce  data/raw/.../BraTS20_Training_001_t1ce.nii \
    --t2    data/raw/.../BraTS20_Training_001_t2.nii \
    --flair data/raw/.../BraTS20_Training_001_flair.nii \
    --ckpt  runs/exp1/best.pt \
    --output runs/exp1/predictions/001/seg.nii.gz \
    --visualize
```

## Repository layout

```
configs/        YAML hyper-parameter files (default.yaml is the reference)
data/raw/       Raw NIfTI volumes — never modified
app/            Streamlit demo app
scripts/        CLI entry points: train.py, evaluate.py, predict.py, verify_setup.py
src/
  models/       Model definitions (unet.py wraps MONAI UNet)
  training/     Trainer, losses, metrics
  inference/    Single-case sliding-window predict
  utils/        Visualisation helpers
runs/           Training artefacts — git-ignored
```

## Demo

### Run locally

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. Upload the four modalities for one
patient (T1, T1ce, T2, FLAIR as `.nii` or `.nii.gz`), optionally upload the
ground-truth segmentation for side-by-side comparison, select a checkpoint from
the dropdown, then click **Run inference**.

**What the UI shows:**

- **Sidebar** — four modality uploaders, optional GT uploader, checkpoint
  selector (all `runs/*/best.pt` files, newest first), device indicator
  (CUDA/CPU)
- **Axial-slice slider** — scrub through all depth slices after inference
- **Image columns** — FLAIR grayscale | Ground truth overlay (if uploaded) |
  Prediction overlay. Colour coding: green = WT, blue = TC, red = ET
- **Tumour volume table** — voxel counts and mm³/cm³ volumes for WT, TC, ET
  computed from the NIfTI voxel spacing
- **Download button** — exports the predicted segmentation as `seg.nii.gz`

Inference time: ~30–60 s on an RTX-class GPU, ~2–3 minutes on CPU.

### Hosting on Hugging Face Spaces (recommended)

Hugging Face Spaces with the **Streamlit SDK** is the cleanest free hosting path:

1. Create a Space at `huggingface.co/new-space`, select **Streamlit** as the SDK.
2. Push the repo (exclude `data/` and `.venv/` via `.gitignore`).
3. Add a `requirements.txt` that lists your pip dependencies.
4. Set the entrypoint in `README.md` YAML front-matter:
   ```yaml
   app_file: app/streamlit_app.py
   ```

**Model checkpoint hosting** — `best.pt` is ~154 MB, which exceeds GitHub's 100 MB
file limit and adds unwanted bloat to the repo history. Two clean options:

- **Hugging Face Hub** (recommended): upload via `huggingface_hub` and load at
  app startup with `hf_hub_download("your-user/lumen", "runs/brats_v1/best.pt")`.
- **GitHub LFS**: `git lfs track "*.pt"` before committing — requires LFS quota
  on your GitHub plan and LFS support on the hosting platform.

Either way, do **not** commit `.pt` files directly to the main repo tree.
