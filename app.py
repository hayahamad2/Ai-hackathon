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

        # heads = Sequential (keys forecast_head.0 / forecast_head.2)
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
    return Y * tgt_std.reshape(1, -1) + tgt_mean.reshape(1, -1)

@torch.no_grad()
def predict(x_std):
    x = torch.tensor(x_std).unsqueeze(0).to(DEVICE)  # [1,T,F]
    f, r = model(x)
    return destandardize(f.squeeze(0).cpu().numpy()), torch.sigmoid(r).item()

def plot_vital(hours_past, values_past, hours_future, values_future, vital, baseline=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hours_past, values_past, marker="o", label="Observed (past)")
    ax.plot(hours_future, values_future, marker="X", label="Forecast (next 6h)")
    ax.plot([hours_past[-1], hours_future[0]], [values_past[-1], values_future[0]], "--")
    if baseline is not None and not (pd.isna(baseline)):
        ax.axhline(float(baseline), linestyle=":", linewidth=2, label="Patient baseline")
    ax.set_title(vital)
    ax.set_xlabel("Hour")
    ax.grid(True)
    ax.legend()
    return fig


# =========================
# UI: Upload CSV
# =========================
st.title("Clinical Sepsis Risk Monitoring System")
st.caption("AI-assisted assessment of patient deterioration based on recent vital signs")


uploaded_csv = st.sidebar.file_uploader("Upload PreprocessedDataset.csv", type=["csv"])
if uploaded_csv is None:
    st.info("Please upload the patient monitoring data file to begin the clinical risk assessment.")
    st.stop()

df = pd.read_csv(uploaded_csv)
df.columns = df.columns.str.strip()
df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
df = df.sort_values(["PatientID_tmp", "Hour"]).reset_index(drop=True)

missing = [c for c in (["PatientID_tmp", "Hour", "SepsisLabel"] + list(FEATURES) + list(VITALS)) if c not in df.columns]
if missing:
    st.error(f"Missing columns in uploaded CSV: {missing}")
    st.stop()

# =========================
# Sidebar selections
# =========================
st.sidebar.title("Patient Overview")
pid = st.sidebar.selectbox("Patient Identifier", df["PatientID_tmp"].unique())
g = df[df["PatientID_tmp"] == pid].sort_values("Hour").reset_index(drop=True)

valid_idx = []
for i in range(len(g)):
    if i >= SEQ_LEN - 1 and i <= len(g) - HORIZON - 1:
        valid_idx.append(i)

if not valid_idx:
    st.warning("This patient does not have enough history/future for prediction. Choose another patient.")
    st.stop()

idx = st.sidebar.selectbox(
    "Assessment Time (Hour)",
    valid_idx,
    format_func=lambda i: f"Hour {int(g.loc[i, 'Hour'])}"
)

# =========================
# Prepare model input
# =========================
X_raw = g.loc[idx-SEQ_LEN+1:idx, FEATURES].values.astype(np.float32)
X = standardize(X_raw)

forecast, risk = predict(X)

hours_past = g.loc[idx-SEQ_LEN+1:idx, "Hour"].values
current_hour = float(hours_past[-1])
hours_future = np.arange(current_hour + 1, current_hour + 1 + HORIZON)

# Baseline values if exist (for vitals plots)
baseline_vals = None
base_cols = [f"base_mean_{v}" for v in VITALS]
if all(c in g.columns for c in base_cols):
    baseline_vals = [float(g.loc[idx, c]) if pd.notna(g.loc[idx, c]) else np.nan for c in base_cols]

# =========================
# Main metrics
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("PatientID", str(pid))
c2.metric("Current Hour", str(int(current_hour)))
c3.metric( f"Predicted Sepsis Risk (next {RISK_HORIZON} hours)",
    f"{risk*100:.1f}%",
    help="Estimated probability of sepsis occurring within the next clinical window")

st.divider()

# =========================
# Plots
# =========================
st.subheader("Recent Vital Signs and Short-Term Projection")
st.caption("Observed measurements over the last 12 hours with model-based projection for the next 6 hours")


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

# ============================================================
# XAI: FULL BLOCK (Doctor-friendly) — compatible with your training
# FEATURES = VITALS + d_* + *_dev
# ============================================================

# ---------- XAI helpers ----------
def pretty_name(f):
    if f.startswith("d_"):
        return f"Trend (Δ) {f[2:]}"
    if f.endswith("_dev"):
        return f"Deviation from baseline: {f[:-4]}"
    return f

def clinical_hint(vital_or_feat):
    mapping = {
        "HR": "HR ↑ may indicate stress response, pain, fever, hypovolemia, or early sepsis.",
        "Resp": "RR ↑ may reflect respiratory distress or metabolic compensation.",
        "O2Sat": "O2Sat ↓ may reflect hypoxia or respiratory compromise.",
        "Temp": "Fever or hypothermia can be associated with infection/sepsis.",
        "MAP": "MAP ↓ may indicate hypotension / shock risk.",
        "Lactate": "Lactate ↑ may suggest poor perfusion."
    }
    base = vital_or_feat.replace("d_", "").replace("_dev", "")
    return mapping.get(base, "Contributes to the model’s detected risk pattern.")

def direction_text(dev):
    if pd.isna(dev):
        return "unknown"
    if abs(dev) < 1e-6:
        return "near baseline"
    return "above baseline" if dev > 0 else "below baseline"

def severity_badge(vital, current, baseline):
    """
    Simplified thresholds for interpretability (NOT a clinical standard).
    Adjust as needed.
    """
    if pd.isna(current) or pd.isna(baseline):
        return "N/A"
    dev = current - baseline

    rules = {
        "HR": (15, 25),
        "Resp": (4, 8),
        "Temp": (0.7, 1.2),
        "MAP": (-10, -20),
        "O2Sat": (-3, -6),
        "Lactate": (0.7, 1.5),
    }
    if vital not in rules:
        return "Info"

    a, b = rules[vital]
    if vital in ["MAP", "O2Sat"]:
        if dev <= b: return "High"
        if dev <= a: return "Moderate"
        return "Mild"
    else:
        if dev >= b: return "High"
        if dev >= a: return "Moderate"
        return "Mild"

def sustained_hours(series, baseline, band=0.0):
    if baseline is None or pd.isna(baseline):
        return None
    dev = series.astype(float) - float(baseline)
    return int((np.abs(dev) > band).sum())

def find_related_vital(feature_name, vitals):
    base = feature_name.replace("d_", "").replace("_dev", "")
    return base if base in vitals else None

def get_val(row, col, default=np.nan):
    return float(row[col]) if (col in row.index and pd.notna(row[col])) else default

def apply_counterfactual_last_point_vitals_dev_d(x_cf_raw, v, FEATURES, last_row, prev_row=None):
    """
    Compatible with your training:
    FEATURES = VITALS + [d_*] + [*_dev]
    It updates: vital, dev, d_ at LAST time step.
    """
    if v not in FEATURES:
        return x_cf_raw, False, f"{v} not in FEATURES"

    j_v = FEATURES.index(v)
    base_mean_col = f"base_mean_{v}"

    if base_mean_col not in last_row.index or pd.isna(last_row[base_mean_col]):
        return x_cf_raw, False, f"Missing {base_mean_col}"

    base_mean = float(last_row[base_mean_col])

    # 1) set vital (raw) at last step to baseline mean
    x_cf_raw[-1, j_v] = base_mean

    # 2) update dev to 0 if exists
    dev_name = f"{v}_dev"
    if dev_name in FEATURES:
        x_cf_raw[-1, FEATURES.index(dev_name)] = 0.0

    # 3) update trend d_ if exists: baseline - previous vital
    d_name = f"d_{v}"
    if d_name in FEATURES:
        if prev_row is not None and v in prev_row.index and pd.notna(prev_row[v]):
            prev_val = float(prev_row[v])
        else:
            prev_val = float(x_cf_raw[-2, j_v]) if x_cf_raw.shape[0] >= 2 else base_mean
        x_cf_raw[-1, FEATURES.index(d_name)] = base_mean - prev_val

    return x_cf_raw, True, "OK"


# ---------- XAI UI ----------
st.subheader("Clinical Decision Explanation")
st.caption("Key clinical factors that contributed to the predicted risk")


if CAPTUM_OK:
    st.markdown("### Data Completeness Check (Last 12 Hours)")

    # ✅ compute missing rate BEFORE using it
    raw_window = g.loc[idx-SEQ_LEN+1:idx, FEATURES].values.astype(np.float32)
    miss_rate = np.isnan(raw_window).mean()

    if miss_rate == 0:
        st.success(
            "All required vital signs were available during the last 12 hours. "
            "Risk assessment is based on complete data."
        )
    elif miss_rate < 0.2:
        st.warning(
            "Some vital signs were missing during the last 12 hours. "
            "Risk assessment should be interpreted with caution."
        )
    else:
        st.error(
            "A significant portion of vital signs data is missing. "
            "Risk assessment reliability is reduced."
        )

    st.divider()


    # Integrated Gradients
    with st.spinner("Computing Integrated Gradients..."):
        ig = IntegratedGradients(RiskOnlyWrapper(model))
        x_t = torch.tensor(X).unsqueeze(0).to(DEVICE)
        attr = ig.attribute(x_t, baselines=torch.zeros_like(x_t), n_steps=64)
        A = np.abs(attr.squeeze(0).detach().cpu().numpy())  # [T,F]

    feat_imp = A.mean(axis=0)
    top_k = 5
    top_idx15 = np.argsort(-feat_imp)[:15]
    top_idx5  = np.argsort(-feat_imp)[:top_k]
    top5_feats  = [FEATURES[i] for i in top_idx5]
    top15_feats = [FEATURES[i] for i in top_idx15]

    # Key moments
    st.markdown("### Critical Time Periods")
    st.caption("Hours during which patient data had the strongest impact on the risk assessment")

    time_imp = A.sum(axis=1)
    top_t = np.argsort(-time_imp)[:3]
    for t in top_t:
        hour = int(hours_past[t])
        st.write(f"**Hour {hour}** — influence score: **{time_imp[t]:.3f}**")

    st.divider()

    # Evidence table
    st.markdown("### Clinical Evidence Summary")
    st.caption("Comparison between patient baseline values and current measurements")

    last_row = g.loc[idx]
    evidence_rows = []
    for vital in VITALS:
        baseline_mean = get_val(last_row, f"base_mean_{vital}", np.nan)
        current_val   = get_val(last_row, vital, np.nan)
        dev_val       = get_val(last_row, f"{vital}_dev", np.nan)
        delta_val     = get_val(last_row, f"d_{vital}", np.nan)

        if pd.isna(dev_val) and pd.notna(current_val) and pd.notna(baseline_mean):
            dev_val = current_val - baseline_mean

        sev = severity_badge(vital, current_val, baseline_mean)
        dur = sustained_hours(g.loc[idx-SEQ_LEN+1:idx, vital], baseline_mean, band=0.0)

        evidence_rows.append({
            "Vital": vital,
            "Baseline(mean)": round(baseline_mean, 2) if pd.notna(baseline_mean) else None,
            "Now": round(current_val, 2) if pd.notna(current_val) else None,
            "Deviation(dev)": round(dev_val, 2) if pd.notna(dev_val) else None,
            "Trend Δ (last hr)": round(delta_val, 2) if pd.notna(delta_val) else None,
            "Sustained (hrs/12h)": f"{dur}/{SEQ_LEN}" if dur is not None else None,
            "Severity": sev
        })
    st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True)

    st.divider()

    # Reason cards
    st.markdown("### Primary Clinical Contributors to Risk")

    window = g.loc[idx-SEQ_LEN+1:idx].copy()

    for f in top5_feats:
        vital = find_related_vital(f, VITALS)
        if vital is not None:
            baseline_mean = get_val(last_row, f"base_mean_{vital}", np.nan)
            current_val   = get_val(last_row, vital, np.nan)
            dev_val       = get_val(last_row, f"{vital}_dev", np.nan)
            delta_val     = get_val(last_row, f"d_{vital}", np.nan)

            if pd.isna(dev_val) and pd.notna(current_val) and pd.notna(baseline_mean):
                dev_val = current_val - baseline_mean

            sev = severity_badge(vital, current_val, baseline_mean)
            dur = sustained_hours(window[vital], baseline_mean, band=0.0)

            dev_phrase = f"{dev_val:+.2f} ({direction_text(dev_val)})" if pd.notna(dev_val) else "N/A"

            st.info(
                f"**{vital}** — {clinical_hint(vital)}\n\n"
                f"- **Now:** {current_val:.2f}  | **Baseline:** {baseline_mean:.2f}\n"
                f"- **Deviation:** {dev_phrase}\n"
                f"- **Trend Δ (last hr):** {delta_val:+.2f}\n"
                f"- **Sustained:** {dur}/{SEQ_LEN} hours\n"
                f"- **Severity:** {sev}\n"
                f"- **Model reliance:** High (top attribution)"
            )
        else:
            st.info(
                f"**{pretty_name(f)}**\n\n"
                f"- Meaning: {clinical_hint(f)}\n"
                f"- Model reliance: High (top attribution)"
            )

    st.divider()

    # What-if (counterfactual)
    st.markdown("### Hypothetical Scenario Analysis")
    st.caption("Estimated effect on risk if a selected vital sign returned to the patient’s baseline")

    v_try = st.selectbox("Select a vital sign to normalize", VITALS, index=0)

    x_cf_raw = g.loc[idx-SEQ_LEN+1:idx, FEATURES].values.astype(np.float32).copy()
    prev_row = g.loc[idx-1] if idx - 1 >= 0 else None

    x_cf_raw, ok, msg = apply_counterfactual_last_point_vitals_dev_d(
        x_cf_raw, v_try, FEATURES, last_row=g.loc[idx], prev_row=prev_row
    )

    if ok:
        x_cf_std = standardize(x_cf_raw)
        _, risk_cf = predict(x_cf_std)

        colA, colB, colC = st.columns(3)
        colA.metric("Current risk", f"{risk*100:.1f}%")
        colB.metric("What-if risk", f"{risk_cf*100:.1f}%")
        colC.metric("Δ risk", f"{(risk_cf-risk)*100:+.1f} pp")

        st.caption("This simulation is intended to support clinical interpretation only and does not constitute a treatment recommendation.")

    else:
        st.warning(f"What-if not applied: {msg}")

    st.divider()

    # Heatmap
    st.markdown("### Feature Contribution Timeline")
    st.caption("Relative contribution of each clinical feature over the assessment window")

    heat = A[:, top_idx15]  # [T,15]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(heat, aspect="auto")
    fig.colorbar(im, ax=ax, label="Importance (Integrated Gradients)")

    ax.set_xticks(range(len(top15_feats)))
    ax.set_xticklabels([pretty_name(f) for f in top15_feats], rotation=60, ha="right")
    ax.set_yticks(range(len(hours_past)))
    ax.set_yticklabels(hours_past.astype(int))

    ax.set_xlabel("Features")
    ax.set_ylabel("Past window hours")
    ax.set_title(f"Why Risk={risk*100:.1f}% ? (Model explanation)")
    fig.tight_layout()
    st.pyplot(fig)

else:
    st.warning("Captum not installed – XAI unavailable.")
