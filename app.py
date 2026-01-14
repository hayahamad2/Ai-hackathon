import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# =========================
# Optional XAI (Captum)
# =========================
try:
    from captum.attr import IntegratedGradients
    CAPTUM_OK = True
except Exception:
    CAPTUM_OK = False

st.set_page_config(page_title="Sepsis Time-Machine Dashboard", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Model artifact path (in repo)
# =========================
CKPT_PATH = "sepsis_gru_multistep_artifact.pt"
if not os.path.exists(CKPT_PATH):
    st.error("❌ Model artifact (.pt) not found in repository. Upload it to the repo root.")
    st.stop()

# =========================
# Model Definition (must match training)
# =========================
class GRUMultiTaskMultiStep(nn.Module):
    def __init__(self, n_features, hidden, horizon, n_targets):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.shared = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.15))
        self.forecast_head = nn.Linear(hidden, horizon * n_targets)
        self.risk_head = nn.Linear(hidden, 1)
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        z = self.shared(last)
        forecast = self.forecast_head(z).view(-1, self.horizon, self.n_targets)
        risk_logit = self.risk_head(z)
        return forecast, risk_logit

class RiskOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        _, r = self.model(x)
        return r

# =========================
# Load model artifact
# =========================
artifact = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

FEATURES = artifact["FEATURES"]
VITALS = artifact["VITALS"]
SEQ_LEN = int(artifact["SEQ_LEN"])
HORIZON = int(artifact["HORIZON"])
RISK_HORIZON = int(artifact["RISK_HORIZON"])

feat_mean = artifact["feat_mean"]
feat_std  = artifact["feat_std"]
tgt_mean  = artifact["tgt_mean"]
tgt_std   = artifact["tgt_std"]

model = GRUMultiTaskMultiStep(
    n_features=len(FEATURES),
    hidden=128,
    horizon=HORIZON,
    n_targets=len(VITALS)
).to(DEVICE)

model.load_state_dict(artifact["model_state"])
model.eval()

# =========================
# Helpers
# =========================
def standardize(X):
    X = (X - feat_mean) / feat_std
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def destandardize(Y):
    return Y * tgt_std.reshape(1,-1) + tgt_mean.reshape(1,-1)

@torch.no_grad()
def predict(x_std):
    x = torch.tensor(x_std).unsqueeze(0).to(DEVICE)
    f, r = model(x)
    return destandardize(f.squeeze(0).cpu().numpy()), torch.sigmoid(r).item()

def plot_vital(hours_past, values_past, hours_future, values_future, vital, baseline=None):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(hours_past, values_past, marker="o", label="Observed (past)")
    ax.plot(hours_future, values_future, marker="X", label="Forecast (next 6h)")
    ax.plot([hours_past[-1], hours_future[0]], [values_past[-1], values_future[0]], "--")
    if baseline is not None:
        ax.axhline(baseline, linestyle=":", linewidth=2, label="Patient baseline")
    ax.set_title(vital)
    ax.set_xlabel("Hour")
    ax.grid(True)
    ax.legend()
    return fig

# =========================
# UI: Upload CSV (instead of keeping it in repo)
# =========================
st.title("🩺 Sepsis Time-Machine Dashboard (GRU + XAI)")
st.caption("Upload your PreprocessedDataset.csv (large files should NOT be stored in GitHub).")

uploaded_csv = st.sidebar.file_uploader("Upload PreprocessedDataset.csv", type=["csv"])
if uploaded_csv is None:
    st.info("⬅️ ارفعي ملف PreprocessedDataset.csv من الشريط الجانبي ثم سيظهر الداشبورد.")
    st.stop()

df = pd.read_csv(uploaded_csv)
df.columns = df.columns.str.strip()
df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
df = df.sort_values(["PatientID_tmp", "Hour"]).reset_index(drop=True)

# Check required columns
missing = [c for c in (["PatientID_tmp","Hour","SepsisLabel"] + FEATURES + VITALS) if c not in df.columns]
if missing:
    st.error(f"Missing columns in uploaded CSV: {missing}")
    st.stop()

# =========================
# Sidebar selections
# =========================
st.sidebar.title("Patient Selection")
pid = st.sidebar.selectbox("PatientID", df["PatientID_tmp"].unique())
g = df[df["PatientID_tmp"] == pid].sort_values("Hour").reset_index(drop=True)

# Valid indices: need history + future
valid_idx = []
for i in range(len(g)):
    if i >= SEQ_LEN-1 and i <= len(g) - HORIZON - 1:
        valid_idx.append(i)

if not valid_idx:
    st.warning("This patient does not have enough history/future for prediction. Choose another patient.")
    st.stop()

idx = st.sidebar.selectbox(
    "Current time (index)",
    valid_idx,
    format_func=lambda i: f"Hour {int(g.loc[i,'Hour'])}"
)

# =========================
# Prepare model input
# =========================
X = standardize(g.loc[idx-SEQ_LEN+1:idx, FEATURES].values.astype(np.float32))
forecast, risk = predict(X)

hours_past = g.loc[idx-SEQ_LEN+1:idx, "Hour"].values
current_hour = float(hours_past[-1])
hours_future = np.arange(current_hour + 1, current_hour + 1 + HORIZON)

# Baseline values if exist
baseline_vals = None
base_cols = [f"base_mean_{v}" for v in VITALS]
if all(c in g.columns for c in base_cols):
    baseline_vals = [float(g.loc[idx, c]) for c in base_cols]

# =========================
# Main metrics
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("PatientID", str(pid))
c2.metric("Current Hour", str(int(current_hour)))
c3.metric(f"Sepsis Risk (next {RISK_HORIZON}h)", f"{risk*100:.1f}%")

st.divider()

# =========================
# Plots
# =========================
st.subheader("⏱️ Observed (last 12h) + Forecast (next 6h)")

row1 = st.columns(3)
row2 = st.columns(3)

for j, vital in enumerate(VITALS):
    fig = plot_vital(
        hours_past,
        g.loc[idx-SEQ_LEN+1:idx, vital].values,
        hours_future,
        forecast[:, j],
        vital,
        baseline_vals[j] if baseline_vals else None
    )
    if j < 3:
        row1[j].pyplot(fig)
    else:
        row2[j-3].pyplot(fig)

st.divider()

# =========================
# XAI (Integrated Gradients)
# =========================
st.subheader("🧠 Explainable AI (XAI) – Why is risk high?")

if not CAPTUM_OK:
    st.warning("Captum not installed in deployment environment. Add 'captum' to requirements.txt.")
else:
    with st.spinner("Computing Integrated Gradients..."):
        ig = IntegratedGradients(RiskOnlyWrapper(model).to(DEVICE))
        x_t = torch.tensor(X).unsqueeze(0).to(DEVICE)
        baseline = torch.zeros_like(x_t)
        attr = ig.attribute(x_t, baselines=baseline, target=0, n_steps=64)

    A = np.abs(attr.squeeze(0).detach().cpu().numpy())  # [T,F]
    feat_imp = A.mean(axis=0)
    top_k = 15
    top_idx = np.argsort(-feat_imp)[:top_k]
    top_feats = [FEATURES[i] for i in top_idx]

    def doctor_text(f):
        if f.startswith("d_"):
            return f"Trend change in {f[2:]}"
        if f.endswith("_dev"):
            return f"Deviation from baseline: {f[:-4]}"
        if f.endswith("_z"):
            return f"Abnormality vs baseline: {f[:-2]}"
        return f

    st.markdown("### ✅ Top-3 Clinical Reasons")
    for f in top_feats[:3]:
        st.write("•", doctor_text(f))

    st.markdown("### 🔍 XAI Heatmap (Top-15 Features × Past 12h)")
    fig, ax = plt.subplots(figsize=(10,4.5))
    im = ax.imshow(A[:, top_idx], aspect="auto")
    fig.colorbar(im, ax=ax, label="Attribution magnitude")

    ax.set_yticks(range(len(hours_past)))
    ax.set_yticklabels(hours_past.astype(int))
    ax.set_xticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=60, ha="right")

    ax.set_xlabel("Top features")
    ax.set_ylabel("Past window hours")
    ax.set_title(f"Integrated Gradients | Risk={risk*100:.1f}%")
    fig.tight_layout()
    st.pyplot(fig)

st.caption("For clinicians: Risk% + Forecast + Top-3 reasons. Heatmap is for deeper inspection.")
