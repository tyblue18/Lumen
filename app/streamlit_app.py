"""Lumen — Brain Tumor Segmentation  (Streamlit demo app)

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import streamlit as st
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predict import predict_volume
from src.models.unet import build_unet
from src.utils.visualize import overlay_segmentation

_ROOT       = Path(__file__).parent.parent
_SAMPLE_DIR = Path(__file__).parent / "sample_data"
_SAMPLE_KEYS = ["t1", "t1ce", "t2", "flair", "seg"]

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lumen — Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS + Google Fonts
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter'
    ':wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown("""<style>
html, body, [class*="st-"], .stApp, p, span, div, label, button {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
h1, h2, h3, h4 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
}
p, .stMarkdown p { font-size: 15px; line-height: 1.6; }
.main .block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 4rem !important;
    max-width: 1380px !important;
}
[data-testid="stSidebar"] > div:first-child {
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #26262b !important;
    border-radius: 12px !important;
    background: #141417 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"] {
    gap: 12px !important;
}
.slice-label {
    font-size: 11px;
    color: #6b7280;
    text-align: center;
    margin-top: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 500;
}
[data-testid="stImage"] img {
    background: #ffffff;
    border-radius: 6px;
}
[data-testid="stSlider"] { padding-bottom: 0.5rem; }
[data-testid="stDownloadButton"] button {
    border: 1px solid #26262b !important;
    background: transparent !important;
    color: #e8e8ea !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #10b981 !important;
    color: #10b981 !important;
}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _gz_suffix(data: bytes) -> str:
    return ".nii.gz" if data[:2] == b"\x1f\x8b" else ".nii"


def _load_nib_bytes(data: bytes):
    """Load a NIfTI from raw bytes; return (fdata float32, affine, header)."""
    suffix = _gz_suffix(data)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        img    = nib.load(tmp_path)
        fdata  = img.get_fdata(dtype=np.float32)
        affine = img.affine.copy()
        header = img.header.copy()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return fdata, affine, header


def _label_to_channels_2d(seg_2d: np.ndarray) -> np.ndarray:
    """Integer label map (H, W) → binary (3, H, W) float32 [TC, WT, ET]."""
    return np.stack([
        ((seg_2d == 1) | (seg_2d == 4)).astype(np.float32),
        ((seg_2d == 1) | (seg_2d == 2) | (seg_2d == 4)).astype(np.float32),
        (seg_2d == 4).astype(np.float32),
    ], axis=0)


def _render_slice(flair_2d: np.ndarray, seg_2d: np.ndarray | None) -> np.ndarray:
    """Return (W, H, 3) uint8 array ready for st.image."""
    if seg_2d is None:
        seg_ch = np.zeros((3, *flair_2d.shape), dtype=np.float32)
        alpha  = 0.0
    else:
        seg_ch = _label_to_channels_2d(seg_2d)
        alpha  = 0.4
    return np.array(overlay_segmentation(flair_2d, seg_ch, alpha=alpha)).transpose(1, 0, 2)


def _vol_table_html(wt: int, tc: int, et: int, vox_mm3: float) -> str:
    """Return a styled HTML table for tumour volumes."""
    TH = ("style='color:#6b7280;font-size:11px;letter-spacing:0.06em;"
          "text-transform:uppercase;padding:10px 16px;font-weight:500;"
          "border-bottom:1px solid #26262b;'")
    def row(dot_color: str, label: str, vox: int) -> str:
        td = "style='padding:11px 16px;color:#e8e8ea;font-variant-numeric:tabular-nums;'"
        td_r = ("style='padding:11px 16px;color:#c9d1d9;"
                "text-align:right;font-variant-numeric:tabular-nums;'")
        dot = (f'<span style="display:inline-block;width:8px;height:8px;'
               f'border-radius:50%;background:{dot_color};margin-right:8px;'
               f'vertical-align:middle;"></span>')
        return (f"<tr style='border-bottom:1px solid rgba(255,255,255,0.04);'>"
                f"<td {td}>{dot}{label}</td>"
                f"<td {td_r}>{vox:,}</td>"
                f"<td {td_r}>{vox * vox_mm3:,.0f}</td>"
                f"<td {td_r}>{vox * vox_mm3 / 1000:.2f}</td>"
                f"</tr>")
    return f"""
<table style="width:100%;border-collapse:collapse;font-size:14px;">
  <thead><tr>
    <th {TH} style="text-align:left;">Region</th>
    <th {TH} style="text-align:right;">Voxels</th>
    <th {TH} style="text-align:right;">Volume (mm³)</th>
    <th {TH} style="text-align:right;">Volume (cm³)</th>
  </tr></thead>
  <tbody>
    {row('#32cd32', 'WT — Whole Tumour',     wt)}
    {row('#1e90ff', 'TC — Tumour Core',      tc)}
    {row('#dc143c', 'ET — Enhancing Tumour', et)}
  </tbody>
</table>"""


# ─────────────────────────────────────────────────────────────────────────────
# Cached: model singleton
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def _load_model(ckpt_path: str, device_str: str):
    cfg    = yaml.safe_load((_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    device = torch.device(device_str)
    model  = build_unet(cfg.get("model")).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("epoch", "?"), ckpt.get("best_dice", float("nan"))


# ─────────────────────────────────────────────────────────────────────────────
# Cached: sliding-window inference (keyed on file MD5s + ckpt path)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_prediction(
    _model: torch.nn.Module,
    _t1: bytes, _t1ce: bytes, _t2: bytes, _flair: bytes,
    t1_hash: str, t1ce_hash: str, t2_hash: str, flair_hash: str,
    ckpt_path: str, device_str: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes, np.ndarray]:
    """Returns (seg, affine, zooms, seg_nii_bytes, flair_vol).
    _-prefixed args are excluded from st.cache_data's hash key."""
    device = torch.device(device_str)
    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        paths  = {
            "t1":    tmpdir / f"t1{_gz_suffix(_t1)}",
            "t1ce":  tmpdir / f"t1ce{_gz_suffix(_t1ce)}",
            "t2":    tmpdir / f"t2{_gz_suffix(_t2)}",
            "flair": tmpdir / f"flair{_gz_suffix(_flair)}",
        }
        paths["t1"].write_bytes(_t1)
        paths["t1ce"].write_bytes(_t1ce)
        paths["t2"].write_bytes(_t2)
        paths["flair"].write_bytes(_flair)

        seg, affine, header = predict_volume(
            t1_path=paths["t1"], t1ce_path=paths["t1ce"],
            t2_path=paths["t2"], flair_path=paths["flair"],
            model=_model, device=device,
        )
        zooms     = np.array(header.get_zooms()[:3], dtype=np.float64)
        flair_vol = nib.load(str(paths["flair"])).get_fdata(dtype=np.float32)
        out       = tmpdir / "seg.nii.gz"
        nib.save(nib.Nifti1Image(seg.astype(np.int16), affine, header), str(out))
        seg_bytes = out.read_bytes()

    return seg, affine, zooms, seg_bytes, flair_vol


@st.cache_data(show_spinner=False)
def _load_sample_bytes() -> dict[str, bytes]:
    """Read sample patient files into a bytes dict; cached for the session."""
    return {k: (_SAMPLE_DIR / f"{k}.nii").read_bytes() for k in _SAMPLE_KEYS}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Lumen brand ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:16px 0 20px 0;">
      <div style="font-size:22px;font-weight:800;letter-spacing:-0.04em;
                  color:#e8e8ea;line-height:1.0;">
        <span style="color:#10b981;">L</span>umen
      </div>
      <div style="font-size:11px;color:#6b7280;margin-top:5px;
                  font-weight:500;letter-spacing:0.04em;text-transform:uppercase;">
        Brain Tumor Segmentation
      </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);
               margin:0 0 20px 0;">
    """, unsafe_allow_html=True)

    # ── Modality uploaders ────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:12px;font-weight:600;color:#a0a0a8;"
        "letter-spacing:0.05em;text-transform:uppercase;margin-bottom:8px;'>"
        "MRI Modalities</p>",
        unsafe_allow_html=True,
    )
    t1_file    = st.file_uploader("T1",    type=["nii", "gz"], key="up_t1")
    t1ce_file  = st.file_uploader("T1ce",  type=["nii", "gz"], key="up_t1ce")
    t2_file    = st.file_uploader("T2",    type=["nii", "gz"], key="up_t2")
    flair_file = st.file_uploader("FLAIR", type=["nii", "gz"], key="up_flair")

    st.divider()
    gt_file = st.file_uploader(
        "Ground truth seg (optional)",
        type=["nii", "gz"],
        key="up_gt",
    )

    st.divider()
    st.markdown(
        "<p style='font-size:12px;font-weight:600;color:#a0a0a8;"
        "letter-spacing:0.05em;text-transform:uppercase;margin-bottom:8px;'>"
        "Checkpoint</p>",
        unsafe_allow_html=True,
    )
    ckpt_files  = sorted(
        _ROOT.glob("runs/*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    ckpt_labels = [str(p.relative_to(_ROOT)) for p in ckpt_files]
    ckpt_paths  = [str(p) for p in ckpt_files]

    if ckpt_labels:
        ckpt_idx = st.selectbox(
            "checkpoint",
            range(len(ckpt_labels)),
            format_func=lambda i: ckpt_labels[i],
            index=0,
            label_visibility="collapsed",
        )
        selected_ckpt = ckpt_paths[ckpt_idx]
    else:
        st.warning("No checkpoints found under runs/*/best.pt")
        selected_ckpt = None

    st.divider()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda":
        st.success(f"CUDA — {torch.cuda.get_device_name(0)}")
    else:
        st.warning("CPU only — inference will take ~2–3 minutes")


# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached singleton — spinner only on first load)
# ─────────────────────────────────────────────────────────────────────────────

if selected_ckpt:
    model, epoch, best_dice = _load_model(selected_ckpt, device_str)
    with st.sidebar:
        st.caption(f"Epoch {epoch}  ·  best Dice {best_dice:.4f}")
else:
    model = None

all_uploaded = all(f is not None for f in (t1_file, t1ce_file, t2_file, flair_file))
_has_sample  = all((_SAMPLE_DIR / f"{k}.nii").exists() for k in _SAMPLE_KEYS)
has_results  = "seg" in st.session_state

# ─────────────────────────────────────────────────────────────────────────────
# Main panel — hero header (always shown)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:4px 0 32px 0;">
  <div style="font-size:54px;font-weight:800;letter-spacing:-0.04em;
              color:#e8e8ea;line-height:1.0;margin-bottom:10px;">
    <span style="color:#10b981;">L</span>umen
  </div>
  <div style="font-size:18px;color:#8b8fa8;font-weight:400;margin-bottom:12px;">
    Brain tumor segmentation from multi-modal MRI.
  </div>
  <div style="font-size:13px;color:#6b7280;display:flex;align-items:center;gap:16px;">
    <span>
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                   background:#32cd32;margin-right:5px;vertical-align:middle;"></span>
      WT &mdash; whole tumour
    </span>
    <span>
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                   background:#1e90ff;margin-right:5px;vertical-align:middle;"></span>
      TC &mdash; tumour core
    </span>
    <span>
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                   background:#dc143c;margin-right:5px;vertical-align:middle;"></span>
      ET &mdash; enhancing tumour
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CTA / action section  (layout depends on whether results exist)
# ─────────────────────────────────────────────────────────────────────────────

sample_trigger = False
run_trigger    = False

if not has_results:
    # ── Empty state: centred hero CTA ────────────────────────────────────────
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 28px 0;">
          <div style="font-size:52px;margin-bottom:14px;line-height:1;">🧠</div>
          <div style="font-size:21px;font-weight:700;color:#e8e8ea;
                      letter-spacing:-0.02em;margin-bottom:8px;">
            Ready to analyze
          </div>
          <div style="font-size:15px;color:#8b8fa8;line-height:1.5;">
            Run Lumen on a built-in sample patient,<br>
            or upload your own MRI scans in the sidebar.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if _has_sample and model is not None:
            sample_trigger = st.button(
                "🧪  Try Lumen with sample patient (BraTS_001)",
                type="primary",
                key="sample_cta",
            )
            st.markdown(
                "<p style='text-align:center;color:#6b7280;font-size:13px;"
                "margin-top:8px;'>No upload needed — ~30 s on GPU, ~3 min on CPU.</p>",
                unsafe_allow_html=True,
            )
        elif not model:
            st.info("Add a checkpoint to `runs/*/best.pt` to enable inference.")

        if all_uploaded and model is not None:
            st.markdown(
                "<p style='text-align:center;color:#6b7280;font-size:13px;"
                "margin:20px 0 8px 0;'>— or —</p>",
                unsafe_allow_html=True,
            )
            run_trigger = st.button(
                "Run inference on uploaded files",
                key="run_cta",
            )
        elif not all_uploaded:
            st.markdown(
                "<p style='text-align:center;color:#6b7280;font-size:13px;"
                "margin-top:20px;'>Or upload your own scans in the sidebar →</p>",
                unsafe_allow_html=True,
            )

else:
    # ── Results mode: compact re-run bar ─────────────────────────────────────
    c1, c2, _ = st.columns([1.4, 1.4, 4])
    with c1:
        if _has_sample and model is not None:
            sample_trigger = st.button(
                "Re-run sample",
                key="sample_rerun",
            )
    with c2:
        run_trigger = st.button(
            "Run inference",
            type="primary",
            disabled=not (all_uploaded and model is not None),
            key="run_rerun",
        )

# ─────────────────────────────────────────────────────────────────────────────
# Execute inference (always at module level — full-width status widget)
# ─────────────────────────────────────────────────────────────────────────────

_inference_ran = False

if sample_trigger and _has_sample and model is not None:
    s = _load_sample_bytes()
    with st.status("Lumen is analyzing 4 modalities...", expanded=True) as _status:
        st.write("Preprocessing MRI volumes and running sliding-window inference…")
        seg, affine, zooms, seg_bytes, flair_vol = _run_prediction(
            model,
            s["t1"], s["t1ce"], s["t2"], s["flair"],
            _md5(s["t1"]), _md5(s["t1ce"]), _md5(s["t2"]), _md5(s["flair"]),
            selected_ckpt, device_str,
        )
        _status.update(label="Segmentation complete!", state="complete", expanded=False)
    st.session_state.update(
        seg=seg, zooms=zooms, seg_bytes=seg_bytes,
        flair_vol=flair_vol, gt_bytes=s["seg"],
    )
    _inference_ran = True

elif run_trigger and all_uploaded and model is not None:
    t1_b    = t1_file.getvalue()
    t1ce_b  = t1ce_file.getvalue()
    t2_b    = t2_file.getvalue()
    flair_b = flair_file.getvalue()
    with st.status("Lumen is analyzing 4 modalities...", expanded=True) as _status:
        st.write("Preprocessing MRI volumes and running sliding-window inference…")
        seg, affine, zooms, seg_bytes, flair_vol = _run_prediction(
            model,
            t1_b, t1ce_b, t2_b, flair_b,
            _md5(t1_b), _md5(t1ce_b), _md5(t2_b), _md5(flair_b),
            selected_ckpt, device_str,
        )
        _status.update(label="Segmentation complete!", state="complete", expanded=False)
    st.session_state.update(
        seg=seg, zooms=zooms, seg_bytes=seg_bytes,
        flair_vol=flair_vol,
        gt_bytes=gt_file.getvalue() if gt_file else None,
    )
    _inference_ran = True

if _inference_ran:
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

if "seg" in st.session_state:
    seg       = st.session_state["seg"]
    zooms     = st.session_state["zooms"]
    seg_bytes = st.session_state["seg_bytes"]
    flair_vol = st.session_state["flair_vol"]
    gt_bytes  = st.session_state.get("gt_bytes")

    # ── Axial-slice slider ────────────────────────────────────────────────────
    depth = seg.shape[2]
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    d = st.slider("Axial slice", min_value=0, max_value=depth - 1, value=depth // 2)

    flair_slice = flair_vol[:, :, d]

    # Decode GT if available
    gt_slice = None
    if gt_bytes is not None:
        try:
            gt_vol, _, _ = _load_nib_bytes(gt_bytes)
            gt_slice = np.round(gt_vol[:, :, d]).astype(np.uint8)
        except Exception as exc:
            st.warning(f"Could not parse ground-truth seg: {exc}")

    # ── Image columns inside a card ───────────────────────────────────────────
    has_gt = gt_slice is not None
    with st.container(border=True):
        cols = st.columns(3 if has_gt else 2)
        with cols[0]:
            st.image(_render_slice(flair_slice, None))
            st.markdown(
                f'<div class="slice-label">FLAIR &nbsp;·&nbsp; slice {d}/{depth-1}</div>',
                unsafe_allow_html=True,
            )
        if has_gt:
            with cols[1]:
                st.image(_render_slice(flair_slice, gt_slice))
                st.markdown(
                    f'<div class="slice-label">Ground truth &nbsp;·&nbsp; slice {d}</div>',
                    unsafe_allow_html=True,
                )
        with cols[-1]:
            st.image(_render_slice(flair_slice, seg[:, :, d]))
            st.markdown(
                f'<div class="slice-label">Prediction &nbsp;·&nbsp; slice {d}</div>',
                unsafe_allow_html=True,
            )

    # ── Tumour volumes ────────────────────────────────────────────────────────
    st.markdown(
        "<h4 style='margin-top:28px;margin-bottom:12px;font-size:15px;"
        "font-weight:600;color:#e8e8ea;letter-spacing:-0.01em;'>Tumour volumes</h4>",
        unsafe_allow_html=True,
    )
    vox_mm3 = float(np.prod(zooms))
    wt_vox  = int(np.sum((seg == 1) | (seg == 2) | (seg == 4)))
    tc_vox  = int(np.sum((seg == 1) | (seg == 4)))
    et_vox  = int(np.sum(seg == 4))

    with st.container(border=True):
        st.markdown(
            _vol_table_html(wt_vox, tc_vox, et_vox, vox_mm3),
            unsafe_allow_html=True,
        )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="Download seg.nii.gz",
        data=seg_bytes,
        file_name="seg.nii.gz",
        mime="application/gzip",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer (always shown)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top:56px;padding-top:24px;
            border-top:1px solid rgba(255,255,255,0.07);
            color:#4b5563;font-size:13px;text-align:center;line-height:2;">
  <strong style="color:#8b8fa8;font-weight:600;">Lumen</strong>
  &nbsp;&middot;&nbsp;
  Trained on BraTS 2020
  &nbsp;&middot;&nbsp;
  Dice 0.88&thinsp;/&thinsp;0.83&thinsp;/&thinsp;0.81 (WT/TC/ET)
  &nbsp;&middot;&nbsp;
  <a href="https://github.com/YOUR-USERNAME/lumen" target="_blank"
     style="color:#10b981;text-decoration:none;font-weight:500;">GitHub</a>
</div>
""", unsafe_allow_html=True)
