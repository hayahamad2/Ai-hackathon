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
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.15)
        )

        # ✅ heads = Sequential (عشان المفاتيح forecast_head.0 / forecast_head.2)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, horizon * n_targets)
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

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
def pretty_name(f):
    # تحويل أسماء الفيتشر لصيغة طبيّة
    if f.startswith("d_"):
        return f"Trend (Δ) {f[2:]}"
    if f.endswith("_dev"):
        return f"Deviation from baseline: {f[:-4]}"
    if f.endswith("_z"):
        return f"Abnormality (Z-score): {f[:-2]}"
    return f

def clinical_hint(f):
    # شرح سريع للطبيب (قابل للتعديل)
    mapping = {
        "HR": "Tachycardia (HR ↑) can be early sign of sepsis",
        "Resp": "Increased RR may indicate respiratory distress / compensation",
        "O2Sat": "Lower O2Sat may indicate hypoxia",
        "Temp": "Fever or hypothermia are sepsis indicators",
        "MAP": "Low MAP can indicate hypotension / shock risk",
        "Lactate": "Rising lactate indicates poor perfusion"
    }
    # استخراج اسم القياس الأساسي
    base = f.replace("d_", "").replace("_dev","").replace("_z","")
    return mapping.get(base, "Contributes to risk pattern")

# ====== XAI: Integrated Gradients ======
st.subheader("🧠 Explainable AI (Clinical Explanation)")

if CAPTUM_OK:
    ig = IntegratedGradients(RiskOnlyWrapper(model))
    x_t = torch.tensor(X).unsqueeze(0).to(DEVICE)
    attr = ig.attribute(x_t, baselines=torch.zeros_like(x_t), n_steps=64)
    A = np.abs(attr.squeeze(0).detach().cpu().numpy())   # [T,F]

    # أهمية كل feature = متوسط الإسناد عبر الزمن
    feat_imp = A.mean(axis=0)
    top_k = 5
    top_idx = np.argsort(-feat_imp)[:15]     # للـ heatmap 15
    top5_idx = np.argsort(-feat_imp)[:top_k] # للشرح 5

    top5_feats = [FEATURES[i] for i in top5_idx]
    top15_feats = [FEATURES[i] for i in top_idx]

    # ====== Clinical Reasons (Top 5) ======
    st.markdown("### ✅ Top Reasons (human-friendly)")
    for f in top5_feats:
        st.write(f"**• {pretty_name(f)}** — {clinical_hint(f)}")

    # ====== Evidence Table (Baseline vs Current) ======
    st.markdown("### 📌 Evidence (Baseline vs Now)")
    evidence_rows = []
    last_row = g.loc[idx]  # current row

    for vital in VITALS:
        base_mean_col = f"base_mean_{vital}"
        d_col = f"d_{vital}"
        dev_col = f"{vital}_dev"

        baseline_mean = float(last_row[base_mean_col]) if base_mean_col in g.columns else np.nan
        current_val = float(last_row[vital]) if vital in g.columns else np.nan
        delta_val = float(last_row[d_col]) if d_col in g.columns else np.nan
        dev_val = float(last_row[dev_col]) if dev_col in g.columns else (current_val - baseline_mean)

        evidence_rows.append({
            "Vital": vital,
            "Baseline(mean)": round(baseline_mean, 2) if pd.notna(baseline_mean) else None,
            "Now": round(current_val, 2) if pd.notna(current_val) else None,
            "Deviation": round(dev_val, 2) if pd.notna(dev_val) else None,
            "Trend Δ (last hr)": round(delta_val, 2) if pd.notna(delta_val) else None
        })

    st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True)

    # ====== Heatmap (Top 15) ======
    st.markdown("### 🔥 Attribution Heatmap (Top 15 Features × Past 12h)")
    heat = A[:, top_idx]  # [T,15]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(heat, aspect="auto")
    fig.colorbar(im, ax=ax, label="Importance (Integrated Gradients)")

    ax.set_xticks(range(len(top15_feats)))
    ax.set_xticklabels([pretty_name(f) for f in top15_feats], rotation=60, ha="right")
    ax.set_yticks(range(len(hours_past)))
    ax.set_yticklabels(hours_past.astype(int))

    ax.set_xlabel("Features (human-friendly)")
    ax.set_ylabel("Past window hours")
    ax.set_title(f"Why Risk={risk*100:.1f}% ? (Model explanation)")
    fig.tight_layout()
    st.pyplot(fig)

else:
    st.warning("Captum not installed – XAI unavailable.")
