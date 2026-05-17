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
    page_icon="🌷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — Blossom Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Nunito:wght@300;400;500;600;700&display=swap');

/* ── Base ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* ── App background ───────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 10% 0%, rgba(255,200,220,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 10%, rgba(230,185,255,0.3) 0%, transparent 55%),
        radial-gradient(ellipse 70% 70% at 50% 100%, rgba(255,214,230,0.25) 0%, transparent 60%),
        linear-gradient(160deg, #fff8fb 0%, #fef0f7 40%, #f5eeff 100%);
    min-height: 100vh;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2a1240 0%, #170a28 100%) !important;
    border-right: 1px solid rgba(200,140,220,0.2) !important;
}
[data-testid="stSidebar"] * { color: #c9a7dd !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {
    color: #f5e6ff !important;
    font-family: 'Cormorant Garamond', serif !important;
}

/* ── Typography ───────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', serif !important;
    color: #3a1a55 !important;
    letter-spacing: -0.01em;
}
p, li { color: #5a3070 !important; }
label { color: #7a4888 !important; font-weight: 500 !important; }

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #e8609c 0%, #c050e0 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 2rem !important;
    box-shadow: 0 6px 20px rgba(224,96,156,0.38) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 12px 28px rgba(192,80,224,0.45) !important;
}

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.62) !important;
    border: 2.5px dashed rgba(232,96,156,0.4) !important;
    border-radius: 24px !important;
    backdrop-filter: blur(12px);
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #e8609c !important;
    background: rgba(255,255,255,0.78) !important;
}
[data-testid="stFileUploader"] * { color: #a060c8 !important; }

/* ── Metrics ──────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid rgba(232,96,156,0.2) !important;
    border-radius: 20px !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: 0 4px 18px rgba(160,90,220,0.1) !important;
    backdrop-filter: blur(10px);
}
[data-testid="metric-container"] label {
    color: #b060d0 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #3a1a55 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.9rem !important;
}

/* ── Progress bar ─────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #e8609c, #c050e0, #9b5de5) !important;
    border-radius: 99px !important;
}
.stProgress > div {
    background: rgba(232,96,156,0.1) !important;
    border-radius: 99px !important;
}

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.65) !important;
    border: 1px solid rgba(232,96,156,0.18) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(10px);
}
[data-testid="stExpander"] summary {
    color: #3a1a55 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

/* ── Sliders ──────────────────────────────────────────────── */
[data-baseweb="slider"] [data-baseweb="thumb"] {
    background: #e8609c !important;
    border-color: #e8609c !important;
    box-shadow: 0 0 0 4px rgba(232,96,156,0.25) !important;
}
[data-baseweb="track-foreground"] {
    background: linear-gradient(90deg, #e8609c, #c050e0) !important;
}
[data-baseweb="track-background"] {
    background: rgba(232,96,156,0.15) !important;
}

/* ── Toggle ───────────────────────────────────────────────── */
[data-baseweb="toggle"] { background: #e8609c !important; }

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.58) !important;
    border-radius: 16px !important;
    padding: 5px !important;
    gap: 4px !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(232,96,156,0.18) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #b060d0 !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #e8609c, #c050e0) !important;
    color: #fff !important;
    box-shadow: 0 3px 12px rgba(232,96,156,0.4) !important;
}

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(232,96,156,0.2) !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,0.7) !important;
}

/* ── Alerts ───────────────────────────────────────────────── */
.stSuccess { background: #fff0f8 !important; border-left-color: #e8609c !important; }
.stError   { background: #fff0f0 !important; }
.stInfo    { background: #f5eeff !important; border-left-color: #c050e0 !important; }

hr { border-color: rgba(232,96,156,0.15) !important; }
.stCaption { color: #b09cc0 !important; font-size: 0.72rem !important; }

/* ── Custom components ────────────────────────────────────── */

/* hero */
.fv-hero { padding: 2.5rem 0 1.5rem; }
.fv-hero-eyebrow {
    font-family: 'Nunito', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #e8609c;
    margin-bottom: 0.5rem;
}
.fv-hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 300;
    color: #3a1a55;
    line-height: 1.0;
    letter-spacing: -0.03em;
    margin-bottom: 0.7rem;
}
.fv-hero-accent {
    background: linear-gradient(120deg, #e8609c 0%, #c050e0 60%, #9b5de5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-style: italic;
}
.fv-hero-sub {
    font-size: 0.9rem;
    color: #9070b0;
    letter-spacing: 0.02em;
    line-height: 1.6;
}

/* pill tags */
.fv-tag {
    display: inline-block;
    background: linear-gradient(135deg, rgba(232,96,156,0.08), rgba(192,80,224,0.08));
    color: #c060b0;
    border: 1px solid rgba(232,96,156,0.22);
    border-radius: 50px;
    padding: 3px 14px;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 6px;
    margin-bottom: 5px;
}

/* glassy card */
.fv-card {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(232,96,156,0.16);
    border-radius: 24px;
    padding: 1.6rem;
    box-shadow: 0 4px 24px rgba(155,93,229,0.09);
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* section label */
.fv-section-label {
    font-family: 'Nunito', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c060b0;
    margin-bottom: 0.5rem;
}

/* stats grid */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin: 0.8rem 0;
}
.stat-item {
    background: linear-gradient(135deg, rgba(253,240,250,0.9), rgba(245,235,255,0.9));
    border: 1px solid rgba(232,96,156,0.14);
    border-radius: 16px;
    padding: 0.65rem 0.9rem;
    text-align: center;
}
.stat-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    color: #3a1a55;
    font-weight: 600;
}
.stat-lbl {
    font-size: 0.58rem;
    color: #b09cc0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
}

/* empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    background: rgba(255,255,255,0.5);
    border: 2.5px dashed rgba(232,96,156,0.22);
    border-radius: 28px;
    backdrop-filter: blur(8px);
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 400;
    color: #7a5090;
}
.empty-sub { font-size: 0.8rem; color: #b09cc0; margin-top: 0.4rem; }

/* device card */
.fv-device-card {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(232,96,156,0.18);
    border-radius: 20px;
    padding: 1rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 18px rgba(155,93,229,0.09);
}
.fv-status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.fv-status-rose  { background: #e8609c; box-shadow: 0 0 8px #e8609c; animation: pulse 2s infinite; }
.fv-status-amber { background: #f59e0b; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:0.6} }

/* options panel */
.options-panel {
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(232,96,156,0.16);
    border-radius: 22px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 2px 16px rgba(155,93,229,0.07);
}
.options-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    color: #3a1a55;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* result badge */
.result-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e8609c, #c050e0);
    color: #fff;
    border-radius: 50px;
    padding: 6px 22px;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(232,96,156,0.4);
    letter-spacing: 0.02em;
}

/* about table */
.about-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-bottom: 0.8rem;
}
.about-table th {
    background: linear-gradient(135deg, rgba(232,96,156,0.1), rgba(192,80,224,0.1));
    color: #3a1a55 !important;
    font-family: 'Nunito', sans-serif;
    font-weight: 700;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.6rem 0.9rem;
    text-align: left;
    border-bottom: 1px solid rgba(232,96,156,0.18);
}
.about-table td {
    padding: 0.55rem 0.9rem;
    color: #5a3070 !important;
    border-bottom: 1px solid rgba(232,96,156,0.08);
}
.about-table tr:last-child td { border-bottom: none; }
.about-table td:first-child { font-weight: 600; color: #8040a0 !important; }

/* divider */
.fv-divider {
    display: flex; align-items: center; gap: 1rem; margin: 1.2rem 0;
}
.fv-divider::before,.fv-divider::after {
    content:''; flex:1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232,96,156,0.25), transparent);
}
.fv-divider-icon { color: #e8609c; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AGE_NAMES  = ["0–2","3–9","10–19","20–29","30–39","40–49","50–59","60–69","70+"]
AGE_ICONS  = ["👶","🧒","🧑","👩","🧔","👩‍🦳","👴","👴","🧓"]
IMG_SIZE   = 224
RANK_ICONS = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣"]

CHART_BG    = "#ffffff"
CHART_FG    = "#3a1a55"
CHART_GRID  = "#f8eef8"
CHART_ROSE  = "#e8609c"
CHART_LILAC = "#c050e0"
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
    ax.text(0, 0.08, f"{conf*100:.1f}%", ha="center", va="center",
            fontsize=20, color=CHART_FG, fontweight="bold")
    ax.text(0, -0.32, "CONFIDENCE", ha="center", fontsize=7, color=CHART_MUTED, fontweight="bold")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.5, 1.1); ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig

def make_distribution_bar(probs):
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    max_idx = int(np.argmax(probs))
    colors  = [CHART_ROSE if i == max_idx else "#e8c0d8" for i in range(len(probs))]
    bars    = ax.bar(AGE_NAMES, probs * 100, color=colors, width=0.65, zorder=3,
                     edgecolor="#ffffff", linewidth=1.2)
    for bar, p in zip(bars, probs * 100):
        if p > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{p:.1f}", ha="center", va="bottom", fontsize=7.5, color=CHART_FG)
    ax.set_ylabel("Probability (%)", color=CHART_MUTED, fontsize=8)
    ax.tick_params(axis="x", rotation=35, colors=CHART_FG, labelsize=8)
    ax.tick_params(axis="y", colors=CHART_MUTED, labelsize=7)
    ax.set_ylim(0, max(probs * 100) * 1.35 + 3)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#f0e6f6")
    ax.set_title("Probability Across All Age Groups", color=CHART_FG, fontsize=9, pad=10, fontweight="bold")
    ax.grid(axis="y", color=CHART_GRID, linewidth=0.8, zorder=0)
    fig.tight_layout()
    return fig

def make_horizontal_bars(top_results):
    n      = len(top_results)
    labels = [f"{r['icon']}  {r['label']}" for r in top_results]
    probs  = [r["prob"] * 100 for r in top_results]
    palette = [CHART_ROSE, CHART_LILAC, "#a784f0"][:n]
    fig, ax = plt.subplots(figsize=(5, max(1.8, n * 0.7 + 0.6)), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    bars = ax.barh(labels[::-1], probs[::-1], color=palette[::-1],
                   height=0.5, edgecolor="#ffffff", linewidth=0.8)
    ax.set_xlim(0, 115)
    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 1.8, bar.get_y() + bar.get_height()/2,
                f"{p:.1f}%", va="center", fontsize=9.5, color=CHART_FG, fontweight="bold")
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
    ax.set_facecolor("#fdf5fb")
    ax.plot(angles, vals, color=CHART_ROSE, linewidth=2.5)
    ax.fill(angles, vals, color=CHART_ROSE, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AGE_NAMES, size=7, color=CHART_FG)
    ax.set_yticklabels([])
    ax.grid(color="#f0e0f0", linewidth=0.8)
    ax.spines["polar"].set_color("#f0e0f0")
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
    ax.text(0, -0.55, "LOW", fontsize=6.5, color="#22c55e", fontweight="bold")
    ax.text(100, -0.55, "HIGH", fontsize=6.5, color="#ef4444", ha="right", fontweight="bold")
    ax.axis("off"); ax.set_ylim(-0.8, 0.6)
    fig.tight_layout(pad=0.1)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  (minimal — collapsed by default)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌷 FairVision")
    model_path = st.text_input("Weights file (.pth)", value="best_fairface_model.pth")
    load_btn   = st.button("🔄 Load / Reload Model", use_container_width=True)
    st.markdown("---")
    st.caption("IJSE · CAME · 2025/2026")
    st.caption("Not for surveillance or biometric ID.")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
if "model_path" not in st.session_state:
    st.session_state.model_path = "best_fairface_model.pth"

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
hcol1, hcol2 = st.columns([3.5, 1])
with hcol1:
    st.markdown("""
<div class="fv-hero">
  <div class="fv-hero-eyebrow">🌷 Age Intelligence · FairFace Dataset</div>
  <div class="fv-hero-title">Fair<span class="fv-hero-accent">Vision</span></div>
  <div class="fv-hero-sub">A CNN-powered age group classifier — elegant, fair, and insightful.</div>
  <div style="margin-top:0.9rem">
    <span class="fv-tag">BetterCNN</span>
    <span class="fv-tag">9 Age Classes</span>
    <span class="fv-tag">PyTorch</span>
    <span class="fv-tag">97,698 Samples</span>
    <span class="fv-tag">FairFace 0.25</span>
  </div>
</div>
""", unsafe_allow_html=True)

with hcol2:
    dev_label = "CUDA ✨" if torch.cuda.is_available() else "CPU"
    dev_dot   = "fv-status-rose" if torch.cuda.is_available() else "fv-status-amber"
    mp_name   = os.path.basename(st.session_state.model_path)
    st.markdown(f"""
<div class="fv-device-card" style="margin-top:2.8rem">
  <div style="font-size:0.62rem;color:#b09cc0;text-transform:uppercase;letter-spacing:0.14em;font-weight:700">Device</div>
  <div style="font-family:'Cormorant Garamond',serif;font-size:1.3rem;color:#3a1a55;margin:0.3rem 0">
    <span class="fv-status-dot {dev_dot}"></span>{dev_label}
  </div>
  <div style="font-size:0.62rem;color:#b09cc0">{mp_name}</div>
  <div style="font-size:0.62rem;color:#e8609c;margin-top:0.3rem;font-weight:700">✅ Ready</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fv-divider"><span class="fv-divider-icon">🌸 🌸 🌸</span></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_infer, tab_about = st.tabs(["🔍  Analyse a Face", "📖  About & Architecture"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:

    # ── Upload + options side by side ─────────────────────────────────────────
    up_col, opt_col = st.columns([1.4, 1], gap="large")

    with up_col:
        st.markdown('<div class="fv-section-label">📷 Upload Your Photo</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a face image here",
            type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed",
        )

    with opt_col:
        st.markdown('<div class="options-panel">', unsafe_allow_html=True)
        st.markdown('<div class="options-title">✨ Inference Options</div>', unsafe_allow_html=True)

        infer_mode = st.radio(
            "Mode",
            ["Standard", "Test-Time Augmentation (TTA)"],
            horizontal=True,
        )
        use_tta = infer_mode == "Test-Time Augmentation (TTA)"

        col_a, col_b = st.columns(2)
        with col_a:
            if use_tta:
                tta_n = st.select_slider("TTA Views", options=[2, 3, 4, 5], value=3)
                temperature = 1.0
            else:
                temperature = st.select_slider(
                    "🌡️ Temperature",
                    options=[0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0],
                    value=1.0,
                )
                tta_n = 3
        with col_b:
            topk = st.select_slider("🏅 Top-K Results", options=list(range(2, 10)), value=3)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Preprocessing tweaks (expandable) ────────────────────────────────────
    with st.expander("🖼️ Image Preprocessing Tweaks  (optional)"):
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            brightness = st.slider("☀️ Brightness", 0.5, 1.8, 1.0, 0.05)
        with pc2:
            contrast   = st.slider("🎛️ Contrast", 0.5, 1.8, 1.0, 0.05)
        with pc3:
            st.write("")
            sharpen = st.toggle("🔪 Sharpen", value=False)
        with pc4:
            st.write("")
            denoise = st.toggle("🧹 Denoise", value=False)

    st.markdown("")

    # ── Results ───────────────────────────────────────────────────────────────
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

        top1    = top_results[0]
        conf    = top1["prob"]
        ent     = entropy(all_probs)
        max_ent = entropy(np.ones(9)/9)
        uncert  = ent / max_ent

        # ── Row 1: image + results ────────────────────────────────────────
        img_col, res_col = st.columns([1, 1.8], gap="large")

        with img_col:
            st.markdown('<div class="fv-card">', unsafe_allow_html=True)
            st.image(proc_img, caption="Your uploaded photo", use_container_width=True)
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
            st.markdown(f"""
<div style="margin-bottom:1.2rem">
  <div class="fv-section-label">✨ Top Prediction</div>
  <div style="margin-top:0.5rem">
    <span class="result-badge">{top1['icon']}  {top1['label']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            conf_label = "High ✅" if conf > 0.6 else "Medium ⚠️" if conf > 0.35 else "Low ❌"
            unc_label  = "Low ✅" if uncert < 0.4 else "High ⚠️"
            m1.metric("Confidence",  f"{conf*100:.1f}%",   delta=conf_label)
            m2.metric("Uncertainty", f"{uncert*100:.1f}%", delta=unc_label,
                      delta_color="normal" if uncert < 0.4 else "inverse")
            m3.metric("Latency",     f"{latency:.0f} ms")

            st.markdown('<div class="fv-divider"><span class="fv-divider-icon">🌸</span></div>', unsafe_allow_html=True)

            g1, g2 = st.columns([1, 1.4])
            with g1:
                st.pyplot(make_gauge(conf), use_container_width=True)
            with g2:
                st.markdown('<div class="fv-section-label" style="margin-bottom:0.3rem">Prediction Uncertainty</div>', unsafe_allow_html=True)
                st.pyplot(make_entropy_bar(ent, max_ent), use_container_width=True)
                st.markdown('<div class="fv-section-label" style="margin-top:0.8rem;margin-bottom:0.4rem">Ranked Predictions</div>', unsafe_allow_html=True)
                for rank, r in enumerate(top_results):
                    icon  = RANK_ICONS[rank]
                    label = f"{icon} {r['icon']} **{r['label']}**"
                    st.progress(float(r["prob"]), text=f"{label} — `{r['prob']*100:.2f}%`")

        # ── Row 2: charts ─────────────────────────────────────────────────
        st.markdown('<div class="fv-divider"><span class="fv-divider-icon">🌸 🌸 🌸</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="fv-section-label">📊 Full Distribution Analysis</div>', unsafe_allow_html=True)

        ch1, ch2, ch3 = st.columns([2.2, 1.5, 1.3])
        with ch1:
            st.pyplot(make_distribution_bar(all_probs), use_container_width=True)
        with ch2:
            st.pyplot(make_horizontal_bars(top_results[:3]), use_container_width=True)
        with ch3:
            st.pyplot(make_radar(all_probs), use_container_width=True)

        # ── Expandable detail ─────────────────────────────────────────────
        with st.expander("🔢 Raw Probability Table — All 9 Age Groups"):
            import pandas as pd
            order  = np.argsort(all_probs)[::-1]
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
  <div class="empty-icon">🌷</div>
  <div class="empty-title">Drop a face photo to reveal its age story</div>
  <div class="empty-sub">Supports JPG · PNG · WEBP &nbsp;·&nbsp; Clear, forward-facing photos work best</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    ac1, ac2 = st.columns(2, gap="large")

    with ac1:
        st.markdown("""
### Project Overview
**FairVision** is a CNN-based age group classifier built for the IJSE Certified AI & ML Engineer programme.
It audits demographic fairness across race and gender using the FairFace dataset.
""")
        st.markdown("#### Architecture — BetterCNN")
        st.markdown("""
<table class="about-table">
<thead><tr><th>Layer</th><th>Detail</th></tr></thead>
<tbody>
<tr><td>Block 1</td><td>Conv 3→32, BN, ReLU ×2, MaxPool</td></tr>
<tr><td>Block 2</td><td>Conv 32→64, BN, ReLU ×2, MaxPool</td></tr>
<tr><td>Block 3</td><td>Conv 64→128, BN, ReLU ×2, MaxPool</td></tr>
<tr><td>Block 4</td><td>Conv 128→256, BN, ReLU ×2, MaxPool</td></tr>
<tr><td>Block 5</td><td>Conv 256→512, BN, ReLU, Dropout2d(0.15)</td></tr>
<tr><td>Pooling</td><td>AdaptiveAvgPool2d(1×1)</td></tr>
<tr><td>Head</td><td>512→256→9, Dropout(0.35)</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)
        st.markdown("""
#### Training Setup
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-5
- **Loss:** CrossEntropyLoss (+ class-weighted for mitigation)
- **Sampler:** WeightedRandomSampler for balanced mini-batches
- **Augmentation:** Flip, rotation, colour jitter · **Input:** 224 × 224
""")

    with ac2:
        st.markdown("#### Dataset — FairFace 0.25")
        st.markdown("""
<table class="about-table">
<thead><tr><th>Split</th><th>Samples</th></tr></thead>
<tbody>
<tr><td>Train (subset)</td><td>29,750</td></tr>
<tr><td>Validation</td><td>5,250</td></tr>
<tr><td>Test</td><td>10,954</td></tr>
<tr><td>Total (public)</td><td>97,698</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)
        st.markdown("#### Fairness Audit Groups")
        st.markdown("""
<table class="about-table">
<thead><tr><th>Attribute</th><th>Role</th><th>Classes</th></tr></thead>
<tbody>
<tr><td>Age</td><td>Target</td><td>9 groups</td></tr>
<tr><td>Race</td><td>Audit</td><td>7 groups</td></tr>
<tr><td>Gender</td><td>Audit</td><td>Male · Female</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)
        st.markdown("#### Inference Features")
        st.markdown("""
<table class="about-table">
<thead><tr><th>Feature</th><th>Description</th></tr></thead>
<tbody>
<tr><td>Temperature</td><td>Scale logits before softmax</td></tr>
<tr><td>TTA</td><td>Average over augmented views</td></tr>
<tr><td>Top-K</td><td>Show 1–9 ranked predictions</td></tr>
<tr><td>Uncertainty</td><td>Entropy-based confidence score</td></tr>
<tr><td>Preprocessing</td><td>Brightness / contrast / sharpen / denoise</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)
        st.info("⚠️ **Limitations** — Age estimation is ambiguous at group boundaries. ~7 pp race performance gap exists. Not suitable for surveillance or biometric ID.")

    st.markdown('<div class="fv-divider"><span class="fv-divider-icon">🌸 🌸 🌸</span></div>', unsafe_allow_html=True)
    st.caption("FairVision · IJSE – Certified AI & ML Engineer · Individual Assignment 2025/2026 · Built with PyTorch & Streamlit 🌷")
