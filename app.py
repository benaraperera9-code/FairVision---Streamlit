import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairVision · Age Intelligence",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(160deg, #fdf6f0 0%, #fce8f3 50%, #ede8f9 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d1b4e 0%, #1a0f30 100%) !important;
    border-right: 1px solid rgba(199,138,210,0.25) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #c9a7dd !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #f5e6ff !important;
    font-family: 'DM Serif Display', serif !important;
}
[data-testid="stSidebar"] .stMarkdown table td,
[data-testid="stSidebar"] .stMarkdown table th {
    color: #d6b8e8 !important;
    border-color: rgba(199,138,210,0.2) !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(199,138,210,0.2) !important; }

/* sidebar inputs */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    color: #f5e6ff !important;
    border: 1px solid rgba(199,138,210,0.3) !important;
    border-radius: 8px !important;
}

/* sidebar slider */
[data-testid="stSidebar"] [data-baseweb="slider"] [data-baseweb="thumb"] {
    background: #e062a0 !important; border-color: #e062a0 !important;
}
[data-testid="stSidebar"] [data-baseweb="track-foreground"] {
    background: linear-gradient(90deg,#e062a0,#9b5de5) !important;
}
[data-testid="stSidebar"] [data-baseweb="track-background"] {
    background: rgba(199,138,210,0.25) !important;
}
[data-testid="stSidebar"] [data-baseweb="toggle"] {
    background: #e062a0 !important;
}

/* ── Main headings ── */
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif !important;
    color: #2d1b4e !important;
}
p, li { color: #4a3060 !important; }
label { color: #6b4474 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(224,98,160,0.18);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 14px rgba(155,93,229,0.08);
    backdrop-filter: blur(8px);
}
[data-testid="metric-container"] label {
    color: #9b5de5 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2d1b4e !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.7rem !important;
    font-weight: 400 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e062a0, #9b5de5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(224,98,160,0.35) !important;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(155,93,229,0.45) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.6) !important;
    border: 2px dashed rgba(224,98,160,0.35) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s !important;
    backdrop-filter: blur(6px);
}
[data-testid="stFileUploader"]:hover { border-color: #e062a0 !important; }
[data-testid="stFileUploader"] * { color: #9b5de5 !important; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #e062a0, #9b5de5) !important;
    border-radius: 99px !important;
}
.stProgress > div {
    background: rgba(224,98,160,0.12) !important;
    border-radius: 99px !important;
}
.stProgress p { color: #2d1b4e !important; font-size: 0.82rem !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.65) !important;
    border: 1px solid rgba(224,98,160,0.18) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(8px);
}
[data-testid="stExpander"] summary {
    color: #2d1b4e !important;
    font-weight: 600 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.55) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(224,98,160,0.15) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #9b5de5 !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#e062a0,#9b5de5) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(224,98,160,0.35) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(224,98,160,0.18) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.7) !important;
}

/* ── Alerts ── */
.stSuccess { background: #fdf0f8 !important; border-color: #e062a0 !important; }
.stError   { background: #fff0f0 !important; border-color: #f87171 !important; }
.stInfo    { background: #f3eeff !important; border-color: #9b5de5 !important; }

hr { border-color: rgba(224,98,160,0.18) !important; }
.stCaption, caption { color: #b09cc0 !important; font-size: 0.72rem !important; }

/* ── Custom components ── */
.fv-hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #2d1b4e;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.fv-hero-accent {
    background: linear-gradient(90deg, #e062a0, #9b5de5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.fv-sub {
    color: #9b5de5;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif;
}
.fv-tag {
    display: inline-block;
    background: rgba(224,98,160,0.1);
    color: #c0397a;
    border: 1px solid rgba(224,98,160,0.25);
    border-radius: 99px;
    padding: 2px 12px;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    font-family: 'DM Sans', sans-serif;
    margin-right: 5px;
    margin-bottom: 4px;
}
.fv-card {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(224,98,160,0.15);
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 2px 16px rgba(155,93,229,0.08);
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.fv-card-accent { border-top: 3px solid #e062a0; }
.fv-status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.fv-status-rose  { background: #e062a0; box-shadow: 0 0 6px #e062a0; }
.fv-status-amber { background: #f59e0b; }
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin: 0.8rem 0;
}
.stat-item {
    background: rgba(253,246,240,0.8);
    border: 1px solid rgba(224,98,160,0.15);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #2d1b4e;
}
.stat-lbl {
    font-size: 0.6rem;
    color: #b09cc0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: rgba(255,255,255,0.55);
    border: 2px dashed rgba(224,98,160,0.25);
    border-radius: 18px;
    backdrop-filter: blur(6px);
}
.empty-icon { font-size: 3rem; margin-bottom: 0.75rem; }
.empty-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #7a5980;
}
.empty-sub { font-size: 0.78rem; color: #b09cc0; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AGE_NAMES  = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","more than 70"]
AGE_ICONS  = ["👶","🧒","🧑","👩","🧔","👩‍🦳","👴","👴","🧓"]
IMG_SIZE   = 224
RANK_ICONS = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣"]

CHART_BG    = "#ffffff"
CHART_FG    = "#2d1b4e"
CHART_GRID  = "#f5eef8"
CHART_ROSE  = "#e062a0"
CHART_LILAC = "#9b5de5"
CHART_MUTED = "#b09cc0"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class BetterCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,256), nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256,num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def apply_preprocessing(img, brightness, contrast, sharpen, denoise):
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BetterCNN(num_classes=9)
    ckpt   = torch.load(path, map_location=device)
    state  = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, device, pil_image, temperature=1.0, topk=3):
    tensor = val_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor) / temperature
        probs  = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=topk)
    results = [
        {"label": AGE_NAMES[i.item()], "icon": AGE_ICONS[i.item()],
         "prob": p.item(), "idx": i.item()}
        for i, p in zip(top_idx, top_probs)
    ]
    return results, probs.cpu().numpy()

def run_tta(model, device, pil_image, n_aug=5):
    tta_transforms = [
        transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE+20,IMG_SIZE+20)),
            transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ColorJitter(brightness=0.1,contrast=0.1), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE+10,IMG_SIZE+10)),
            transforms.RandomCrop(IMG_SIZE), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    ]
    all_probs = []
    with torch.no_grad():
        for t in tta_transforms[:n_aug]:
            tensor = t(pil_image).unsqueeze(0).to(device)
            all_probs.append(F.softmax(model(tensor), dim=1)[0].cpu().numpy())
    avg      = np.mean(all_probs, axis=0)
    top3_idx = avg.argsort()[::-1][:3]
    results  = [{"label": AGE_NAMES[i], "icon": AGE_ICONS[i],
                 "prob": float(avg[i]), "idx": int(i)} for i in top3_idx]
    return results, avg

def entropy(probs):
    p = probs + 1e-9
    return float(-np.sum(p * np.log(p)))

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def make_gauge(conf: float):
    fig, ax = plt.subplots(figsize=(3.2, 2.0), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    theta      = np.linspace(np.pi, 0, 300)
    fill_theta = np.linspace(np.pi, np.pi - conf * np.pi, 300)
    color = CHART_ROSE if conf > 0.6 else "#f59e0b" if conf > 0.35 else "#ef4444"
    ax.plot(np.cos(theta), np.sin(theta), color="#f5eef8", linewidth=16, solid_capstyle="round")
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=16, solid_capstyle="round")
    ax.text(0, 0.05, f"{conf*100:.1f}%", ha="center", va="center",
            fontsize=20, color=CHART_FG, fontweight="bold")
    ax.text(0, -0.35, "CONFIDENCE", ha="center", fontsize=7, color=CHART_MUTED)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.5, 1.1); ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig

def make_distribution_bar(probs):
    fig, ax = plt.subplots(figsize=(8, 3.0), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    max_idx = int(np.argmax(probs))
    colors  = [CHART_ROSE if i == max_idx else "#e9d8f7" for i in range(len(probs))]
    bars    = ax.bar(AGE_NAMES, probs * 100, color=colors, width=0.6, zorder=3,
                     edgecolor="#f5eef8", linewidth=0.8)
    for bar, p in zip(bars, probs * 100):
        if p > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{p:.1f}", ha="center", va="bottom", fontsize=7.5, color=CHART_FG)
    ax.set_ylabel("Probability (%)", color=CHART_MUTED, fontsize=8)
    ax.tick_params(axis="x", rotation=35, colors=CHART_FG, labelsize=8)
    ax.tick_params(axis="y", colors=CHART_MUTED, labelsize=7)
    ax.set_ylim(0, max(probs * 100) * 1.3 + 3)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#f0e6f6")
    ax.set_title("Full Probability Distribution", color=CHART_FG, fontsize=9, pad=10, fontweight="bold")
    ax.grid(axis="y", color=CHART_GRID, linewidth=0.8, zorder=0)
    fig.tight_layout()
    return fig

def make_horizontal_bars(top_results):
    n      = len(top_results)
    labels = [r["label"] for r in top_results]
    probs  = [r["prob"] * 100 for r in top_results]
    palette = [CHART_ROSE, CHART_LILAC, "#c084fc"]
    fig, ax = plt.subplots(figsize=(5, max(1.6, n * 0.55 + 0.5)), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    bars = ax.barh(labels[::-1], probs[::-1], color=palette[:n][::-1],
                   height=0.5, edgecolor="#f5eef8", linewidth=0.6)
    ax.set_xlim(0, 110)
    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f"{p:.1f}%", va="center", fontsize=9, color=CHART_FG, fontweight="bold")
    ax.spines[["top","right","left","bottom"]].set_color("#f0e6f6")
    ax.tick_params(colors=CHART_FG, labelsize=9)
    ax.set_xlabel("Confidence (%)", color=CHART_MUTED, fontsize=8)
    ax.grid(axis="x", color=CHART_GRID, linewidth=0.6)
    ax.set_title("Top Predictions", color=CHART_FG, fontsize=9, pad=8, fontweight="bold")
    fig.tight_layout()
    return fig

def make_radar(probs):
    N      = len(AGE_NAMES)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals   = probs.tolist()
    angles += angles[:1]; vals += vals[:1]
    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True), facecolor=CHART_BG)
    ax.set_facecolor("#fdf6fb")
    ax.plot(angles, vals, color=CHART_ROSE, linewidth=2.5)
    ax.fill(angles, vals, color=CHART_ROSE, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AGE_NAMES, size=7, color=CHART_FG)
    ax.set_yticklabels([])
    ax.grid(color="#f0e6f6", linewidth=0.8)
    ax.spines["polar"].set_color("#f0e6f6")
    ax.set_title("Radar View", color=CHART_FG, fontsize=9, pad=14, fontweight="bold")
    fig.tight_layout()
    return fig

def make_entropy_bar(ent_val, max_ent):
    fig, ax = plt.subplots(figsize=(4, 0.9), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    ratio = ent_val / max_ent
    color = "#22c55e" if ratio < 0.35 else "#f59e0b" if ratio < 0.65 else "#ef4444"
    ax.barh([0], [100], color="#f5eef8", height=0.5)
    ax.barh([0], [ratio*100], color=color, height=0.5)
    ax.set_xlim(0, 100)
    ax.text(ratio*100 + 1, 0, f"{ratio*100:.1f}%", va="center",
            fontsize=9, color=CHART_FG, fontweight="bold")
    ax.text(0, -0.55, "LOW UNCERTAINTY", fontsize=6.5, color="#22c55e")
    ax.text(100, -0.55, "HIGH UNCERTAINTY", fontsize=6.5, color="#ef4444", ha="right")
    ax.axis("off"); ax.set_ylim(-0.8, 0.6)
    fig.tight_layout(pad=0.1)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'DM Serif Display\',serif;font-size:1.5rem;color:#f5e6ff;margin-bottom:0">🌸 FairVision</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:0.7rem;color:#9b72c0;letter-spacing:0.12em;text-transform:uppercase;margin-top:0">Age Group Intelligence</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("### ⚙️ Model")
    model_path = st.text_input("Weights file (.pth)", value="best_fairface_model.pth")
    load_btn   = st.button("🔄 Load / Reload Model", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔬 Inference")
    use_tta     = st.toggle("Test-Time Augmentation", value=False,
                            help="Averages predictions over multiple augmented views.")
    tta_n       = st.slider("TTA views", 2, 5, 3, disabled=not use_tta)
    temperature = st.slider("Temperature", 0.3, 2.0, 1.0, 0.1,
                            help="<1 = sharper  |  >1 = smoother")
    topk        = st.slider("Top-K results", 2, 9, 3)

    st.markdown("---")
    st.markdown("### 🖼️ Preprocessing")
    brightness = st.slider("Brightness", 0.5, 1.8, 1.0, 0.05)
    contrast   = st.slider("Contrast",   0.5, 1.8, 1.0, 0.05)
    sharpen    = st.toggle("Sharpen", value=False)
    denoise    = st.toggle("Denoise", value=False)

    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.markdown("""
| | |
|---|---|
| **Arch** | BetterCNN |
| **Input** | 224 × 224 |
| **Classes** | 9 age groups |
| **Framework** | PyTorch |
| **Dataset** | FairFace 0.25 |
""")
    st.markdown("---")
    st.markdown("### ⚠️ Responsible Use")
    st.caption("Not for surveillance, biometric ID, or high-stakes decisions. Performance varies across demographic groups.")
    st.markdown("---")
    st.caption("IJSE · CAME · 2025/2026")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
if "model" not in st.session_state or load_btn:
    if not os.path.isfile(model_path):
        st.error(f"❌ `{model_path}` not found. Place your `.pth` file next to `app.py`.")
        st.stop()
    with st.spinner("Loading model …"):
        try:
            st.session_state.model, st.session_state.device = load_model(model_path)
            st.session_state.model_path = model_path
        except Exception as e:
            st.error(f"Load error: {e}")
            st.stop()

model  = st.session_state.model
device = st.session_state.device

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown(
        '<div class="fv-hero-title">Fair<span class="fv-hero-accent">Vision</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="fv-sub" style="margin:0.4rem 0 0.8rem">CNN · Age Group Intelligence · FairFace Dataset</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
<span class="fv-tag">BetterCNN</span>
<span class="fv-tag">9 Age Classes</span>
<span class="fv-tag">PyTorch</span>
<span class="fv-tag">FairFace 0.25</span>
<span class="fv-tag">97,698 Samples</span>
""", unsafe_allow_html=True)

with hcol2:
    dev_label = "CUDA ✨" if torch.cuda.is_available() else "CPU"
    dev_dot   = "fv-status-rose" if torch.cuda.is_available() else "fv-status-amber"
    st.markdown(f"""
<div class="fv-card" style="text-align:center;padding:1rem;margin-top:0.2rem">
  <div style="font-size:0.7rem;color:#b09cc0;text-transform:uppercase;letter-spacing:0.1em">Device</div>
  <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#2d1b4e;margin:0.2rem 0">
    <span class="fv-status-dot {dev_dot}"></span>{dev_label}
  </div>
  <div style="font-size:0.65rem;color:#b09cc0;margin-top:0.4rem">{os.path.basename(model_path)}</div>
  <div style="font-size:0.6rem;color:#e062a0">✅ Model ready</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_infer, tab_batch, tab_about = st.tabs(
    ["🔍  Single Inference", "📂  Batch Analysis", "📖  About & Architecture"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    up_col, _ = st.columns([2, 1])
    with up_col:
        uploaded = st.file_uploader(
            "Upload a face image (JPG · PNG · WEBP)",
            type=["jpg","jpeg","png","webp"],
        )

    if uploaded:
        raw_img  = Image.open(uploaded).convert("RGB")
        proc_img = apply_preprocessing(raw_img, brightness, contrast, sharpen, denoise)

        t0 = time.perf_counter()
        if use_tta:
            top_results, all_probs = run_tta(model, device, proc_img, n_aug=tta_n)
        else:
            top_results, all_probs = predict(model, device, proc_img,
                                             temperature=temperature, topk=topk)
        latency = (time.perf_counter() - t0) * 1000

        top1   = top_results[0]
        conf   = top1["prob"]
        ent    = entropy(all_probs)
        max_ent= entropy(np.ones(9)/9)
        uncert = ent / max_ent

        # ── Row 1: image + summary ────────────────────────────────────────────
        img_col, res_col = st.columns([1, 1.7], gap="large")

        with img_col:
            st.markdown('<div class="fv-card fv-card-accent">', unsafe_allow_html=True)
            st.image(proc_img, caption="Input (preprocessed)", use_container_width=True)
            w, h = proc_img.size
            st.markdown(f"""
<div class="stat-grid">
  <div class="stat-item"><div class="stat-val">{w}×{h}</div><div class="stat-lbl">Size</div></div>
  <div class="stat-item"><div class="stat-val">{"TTA" if use_tta else "STD"}</div><div class="stat-lbl">Mode</div></div>
  <div class="stat-item"><div class="stat-val">{latency:.0f}ms</div><div class="stat-lbl">Latency</div></div>
  <div class="stat-item"><div class="stat-val">{temperature:.1f}</div><div class="stat-lbl">Temp</div></div>
</div>
""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col:
            g_col, m_col = st.columns([1, 1.3])
            with g_col:
                st.pyplot(make_gauge(conf), use_container_width=True)
            with m_col:
                st.metric("🏆 Top Prediction", f"{top1['icon']} {top1['label']}")
                conf_label = "High ✅" if conf > 0.6 else "Medium ⚠️" if conf > 0.35 else "Low ❌"
                st.metric("Confidence", f"{conf*100:.1f}%", delta=conf_label)
                unc_label  = "Low ✅" if uncert < 0.4 else "High ⚠️"
                st.metric("Uncertainty", f"{uncert*100:.1f}%", delta=unc_label,
                          delta_color="normal" if uncert < 0.4 else "inverse")

            st.markdown("---")
            st.markdown("**Prediction Uncertainty**")
            st.pyplot(make_entropy_bar(ent, max_ent), use_container_width=True)

            st.markdown("---")
            st.markdown(f"**Top-{len(top_results)} Predictions**")
            for rank, r in enumerate(top_results):
                icon  = RANK_ICONS[rank]
                label = f"{icon} {r['icon']} **{r['label']}**"
                st.progress(float(r["prob"]), text=f"{label} — `{r['prob']*100:.2f}%`")

        # ── Row 2: charts ─────────────────────────────────────────────────────
        st.markdown("---")
        ch1, ch2, ch3 = st.columns([2.2, 1.5, 1.3])
        with ch1:
            st.pyplot(make_distribution_bar(all_probs), use_container_width=True)
        with ch2:
            st.pyplot(make_horizontal_bars(top_results[:3]), use_container_width=True)
        with ch3:
            st.pyplot(make_radar(all_probs), use_container_width=True)

        # ── Expanders ─────────────────────────────────────────────────────────
        with st.expander("🔢 Raw Probability Table"):
            import pandas as pd
            order = np.argsort(all_probs)[::-1]
            df_raw = pd.DataFrame({
                "Rank":            [int(np.where(order == i)[0][0]+1) for i in range(9)],
                "Age Group":       AGE_NAMES,
                "Icon":            AGE_ICONS,
                "Probability (%)": [f"{p*100:.4f}" for p in all_probs],
            }).sort_values("Rank").reset_index(drop=True)
            st.dataframe(df_raw, use_container_width=True, hide_index=True)

        if brightness != 1.0 or contrast != 1.0 or sharpen or denoise:
            with st.expander("🖼️ Original vs Preprocessed"):
                c1, c2 = st.columns(2)
                c1.image(raw_img,  caption="Original",     use_container_width=True)
                c2.image(proc_img, caption="Preprocessed", use_container_width=True)

    else:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">🌸</div>
  <div class="empty-title">Upload a face image to begin</div>
  <div class="empty-sub">Supports JPG · PNG · WEBP · Clear forward-facing photos work best</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("Upload **multiple** face images for bulk inference and summary statistics.")

    batch_files = st.file_uploader(
        "Drop multiple images",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        key="batch_up",
    )

    if batch_files:
        if st.button("▶ Run Batch Inference", use_container_width=False):
            import pandas as pd
            results_list = []
            prog = st.progress(0, text="Initialising …")

            for idx, f in enumerate(batch_files):
                img = Image.open(f).convert("RGB")
                img = apply_preprocessing(img, brightness, contrast, sharpen, denoise)
                t0  = time.perf_counter()
                top, probs = predict(model, device, img, temperature=temperature, topk=3)
                lat = (time.perf_counter() - t0) * 1000
                results_list.append({
                    "File":          f.name,
                    "Top-1":         top[0]["label"],
                    "Conf (%)":      f"{top[0]['prob']*100:.2f}",
                    "Top-2":         top[1]["label"],
                    "Top-3":         top[2]["label"],
                    "Entropy":       f"{entropy(probs):.3f}",
                    "Latency (ms)":  f"{lat:.1f}",
                })
                prog.progress((idx+1)/len(batch_files), text=f"Processing {f.name} …")

            prog.empty()
            df = pd.DataFrame(results_list)
            st.success(f"✅ Processed **{len(batch_files)}** images")
            st.dataframe(df, use_container_width=True, hide_index=True)

            from collections import Counter
            top1_counts = Counter(df["Top-1"])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Images Processed", len(batch_files))
            mc2.metric("Most Predicted",   top1_counts.most_common(1)[0][0])
            mc3.metric("Unique Classes",   len(top1_counts))

            # Pie chart
            fig_p, ax_p = plt.subplots(figsize=(5, 4), facecolor=CHART_BG)
            ax_p.set_facecolor(CHART_BG)
            rose_cmap = ["#e062a0","#9b5de5","#c084fc","#f472b6",
                         "#fb7185","#a78bfa","#f0abfc","#fda4af"]
            colors_pie = rose_cmap[:len(top1_counts)]
            wedges, texts, autotexts = ax_p.pie(
                top1_counts.values(), labels=top1_counts.keys(),
                colors=colors_pie, autopct="%1.0f%%", startangle=90,
                wedgeprops=dict(edgecolor="#ffffff", linewidth=1.5)
            )
            for t in texts:     t.set_color(CHART_FG); t.set_fontsize(8)
            for t in autotexts: t.set_color("#ffffff"); t.set_fontsize(8); t.set_fontweight("bold")
            ax_p.set_title("Batch Top-1 Distribution", color=CHART_FG,
                           fontsize=10, fontweight="bold", pad=12)
            st.pyplot(fig_p, use_container_width=False)
    else:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">📂</div>
  <div class="empty-title">Upload multiple images for batch processing</div>
  <div class="empty-sub">Get summary stats, latency reports, and distribution charts for your dataset</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    ac1, ac2 = st.columns(2, gap="large")

    with ac1:
        st.markdown("""
### Project Overview
**FairVision** is a CNN-based age group classification system developed for the
IJSE Certified AI & ML Engineer programme. It audits demographic fairness across
race and gender groups using the FairFace dataset.

### Architecture — BetterCNN
| Layer | Detail |
|---|---|
| Block 1 | Conv 3→32, BN, ReLU ×2, MaxPool |
| Block 2 | Conv 32→64, BN, ReLU ×2, MaxPool |
| Block 3 | Conv 64→128, BN, ReLU ×2, MaxPool |
| Block 4 | Conv 128→256, BN, ReLU ×2, MaxPool |
| Block 5 | Conv 256→512, BN, ReLU, Dropout2d(0.15) |
| Pooling | AdaptiveAvgPool2d(1×1) |
| Head | 512→256→9, Dropout(0.35) |

### Training Setup
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-5
- **Loss:** CrossEntropyLoss (+ class-weighted variant for mitigation)
- **Sampler:** WeightedRandomSampler for balanced mini-batches
- **Augmentation:** Flip, rotation, colour jitter
- **Input size:** 224 × 224
""")

    with ac2:
        st.markdown("""
### Dataset — FairFace 0.25
| Split | Samples |
|---|---|
| Train (subset) | 29,750 |
| Validation | 5,250 |
| Test | 10,954 |
| **Total (public)** | **97,698** |

### Fairness Audit Groups
| Attribute | Role | Classes |
|---|---|---|
| Age | Target | 9 groups |
| Race | Audit | 7 groups |
| Gender | Audit | Male · Female |

### Inference Features
| Feature | Description |
|---|---|
| Temperature | Scale logits before softmax |
| TTA | Average over augmented views |
| Top-K | Show 1–9 ranked predictions |
| Uncertainty | Entropy-based confidence score |
| Preprocessing | Brightness / contrast / sharpen / denoise |
| Batch mode | Multi-image bulk inference |

### Limitations
- Age estimation is ambiguous at group boundaries
- Demographic performance gaps exist (~7 pp race gap)
- Not suitable for surveillance or biometric ID
""")

    st.markdown("---")
    st.caption("FairVision · IJSE – Certified AI & ML Engineer · Individual Assignment 2025/2026 · Built with PyTorch & Streamlit 🌸")
