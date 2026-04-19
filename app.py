"""
app.py
Streamlit web app — Sound Classifier
Based on: "A Multimodal Prototypical Approach for Unsupervised Sound Classification"
(INTERSPEECH 2023) — Proto-LC model (LAION-CLAP + prototypical)
"""

import os
import sys
import io
import time
import tempfile
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ─────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sound Classifier | Proto-LC",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Paths (relative to this file)
# ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "data", "input", "630k-audioset-fusion-best.pt")
PROTOTYPE_PATH  = os.path.join(BASE_DIR, "data", "demo", "mean_embd_tensor_esc50_clap_zs.pt")
LABEL_CSV_PATH  = os.path.join(BASE_DIR, "data", "labels", "esc50.csv")
TEMP_DIR        = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 10
TOP_N = 10

# ─────────────────────────────────────────────────────────────
# Custom CSS — dark, premium theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 50%, #f78166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
}

.main-subtitle {
    font-size: 1rem;
    color: #8b949e;
    font-weight: 400;
    letter-spacing: 0.5px;
}

.paper-badge {
    display: inline-block;
    background: rgba(88, 166, 255, 0.1);
    border: 1px solid rgba(88, 166, 255, 0.3);
    color: #58a6ff;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.5rem;
    letter-spacing: 0.5px;
}

/* ── Cards ── */
.glass-card {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}

/* ── Upload zone ── */
.stFileUploader > div {
    background: rgba(22, 27, 34, 0.6) !important;
    border: 2px dashed rgba(88, 166, 255, 0.4) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease;
}
.stFileUploader > div:hover {
    border-color: rgba(88, 166, 255, 0.8) !important;
}

/* ── Classify button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.3px;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
    box-shadow: 0 4px 20px rgba(88, 166, 255, 0.3) !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 26px rgba(88, 166, 255, 0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-summary {
    background: linear-gradient(135deg, rgba(88,166,255,0.08) 0%, rgba(163,113,247,0.08) 100%);
    border: 1px solid rgba(88, 166, 255, 0.25);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    text-align: center;
}

.result-rank-icon { font-size: 2.5rem; margin-bottom: 0.25rem; }
.result-label { font-size: 2rem; font-weight: 700; color: #e6edf3; }
.result-score { font-size: 1.2rem; color: #58a6ff; font-weight: 600; }

.category-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
    letter-spacing: 0.3px;
}

/* ── Setup status ── */
.status-ok   { color: #3fb950; font-weight: 600; }
.status-fail { color: #f85149; font-weight: 600; }

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
}

/* ── Divider ── */
hr { border-color: rgba(48,54,61,0.6) !important; }

/* ── Streamlit overrides ── */
.stAlert { border-radius: 10px !important; }
[data-testid="stMarkdownContainer"] p { color: #c9d1d9; }
.stAudio { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Cached model loader (only runs once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load all model components. Cached so it only runs once."""
    from classifier import SoundClassifier
    clf = SoundClassifier(MODEL_PATH, PROTOTYPE_PATH, LABEL_CSV_PATH)
    clf.load_model()
    clf.load_prototypes()
    clf.load_labels()
    return clf

def check_setup() -> dict:
    """Check if required files exist before loading model."""
    return {
        "LAION-CLAP model (~600 MB)": (MODEL_PATH, os.path.exists(MODEL_PATH)),
        "Prototype embeddings": (PROTOTYPE_PATH, os.path.exists(PROTOTYPE_PATH)),
        "ESC-50 CSV label map": (LABEL_CSV_PATH, os.path.exists(LABEL_CSV_PATH)),
    }

# ─────────────────────────────────────────────────────────────
# Color helpers
# ─────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Animals":              "#f78166",
    "Natural soundscapes":  "#56d364",
    "Human non-speech":     "#ffa657",
    "Interior/domestic":    "#58a6ff",
    "Exterior/urban":       "#a371f7",
    "Unknown":              "#8b949e",
}

RANK_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

def get_bar_color(rank: int, total: int) -> str:
    """Linear gradient from vivid gold (#FFD700) → desat blue-gray for lower ranks."""
    ratio = (rank - 1) / max(total - 1, 1)
    # Gold → teal-blue
    r = int(255 * (1 - ratio) + 32 * ratio)
    g = int(215 * (1 - ratio) + 139 * ratio)
    b = int(0   * (1 - ratio) + 255 * ratio)
    return f"rgb({r},{g},{b})"

# ─────────────────────────────────────────────────────────────
# Build Plotly chart
# ─────────────────────────────────────────────────────────────
def build_bar_chart(results: list) -> go.Figure:
    labels    = [r["label"].title() for r in results]
    scores    = [r["score"] * 100      for r in results]
    categories= [r["category"]         for r in results]
    ranks     = [r["rank"]             for r in results]

    colors = [get_bar_color(r, len(results)) for r in ranks]
    medal_labels = [
        f"{RANK_MEDALS.get(r, '')} {lbl}" for r, lbl in zip(ranks, labels)
    ]

    hover_text = [
        f"<b>{lbl}</b><br>Rank #{r}<br>Score: {s:.1f}%<br>Category: {cat}"
        for lbl, r, s, cat in zip(labels, ranks, scores, categories)
    ]

    # Reverse for horizontal bar (highest at top)
    fig = go.Figure(go.Bar(
        x=scores[::-1],
        y=medal_labels[::-1],
        orientation="h",
        marker=dict(
            color=colors[::-1],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text[::-1],
        text=[f"{s:.1f}%" for s in scores[::-1]],
        textposition="outside",
        textfont=dict(color="#e6edf3", size=12, family="Inter"),
        cliponaxis=False,
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(
            range=[0, 115],
            ticksuffix="%",
            gridcolor="rgba(48,54,61,0.5)",
            gridwidth=1,
            zeroline=False,
            color="#8b949e",
        ),
        yaxis=dict(
            tickfont=dict(size=13, color="#e6edf3"),
            automargin=True,
        ),
        margin=dict(l=10, r=60, t=20, b=20),
        height=420,
        hoverlabel=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(family="Inter", color="#e6edf3"),
        ),
    )

    return fig

# ─────────────────────────────────────────────────────────────
# Render setup status panel
# ─────────────────────────────────────────────────────────────
def render_setup_status(status: dict):
    all_ok = all(exists for _, (_, exists) in status.items())
    if all_ok:
        st.success("✅ All required files found. Model is ready to load.")
    else:
        st.error("⚠️ Some required files are missing. See setup guide below.")

    for name, (path, exists) in status.items():
        icon = "✅" if exists else "❌"
        rel  = os.path.relpath(path, BASE_DIR)
        cls  = "status-ok" if exists else "status-fail"
        st.markdown(
            f'<span class="{cls}">{icon} **{name}**</span>  \n'
            f'<code style="color:#8b949e;font-size:0.78rem;">{rel}</code>',
            unsafe_allow_html=True,
        )

    if not all_ok:
        with st.expander("📖 Setup Guide — Missing Files"):
            st.markdown("""
### 1. Download LAION-CLAP model (~600 MB)
```bash
# From project root:
Invoke-WebRequest -Uri "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt" `
    -OutFile "data/input/630k-audioset-fusion-best.pt"
```
Or download manually from:  
👉 https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt

---

### 2. Clone ESC-50 dataset (for label CSV)
```bash
cd data/input
git clone https://github.com/karolpiczak/ESC-50.git
```
Required file: `data/input/ESC-50/meta/esc50.csv`

---

### 3. Install dependencies
```bash
pip install streamlit laion-clap torch torchaudio librosa soundfile plotly pandas numpy
```

---

### 4. Run the app
```bash
streamlit run app.py
```
""")

# ─────────────────────────────────────────────────────────────
# Main app layout
# ─────────────────────────────────────────────────────────────
def main():
    # ── Header ─────────────────────────────
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🔊 Sound Classifier</div>
        <div class="main-subtitle">
            Multimodal Prototypical Approach for Unsupervised Sound Classification
        </div>
        <span class="paper-badge">INTERSPEECH 2023 · Proto-LC · LAION-CLAP</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Setup status check ──────────────────
    setup_status = check_setup()
    all_ok = all(exists for _, (_, exists) in setup_status.items())

    with st.expander("🔧 Setup Status", expanded=not all_ok):
        render_setup_status(setup_status)

    if not all_ok:
        st.stop()

    # ── Load model (cached) ─────────────────
    with st.spinner("⏳ Loading LAION-CLAP model and prototypes (first time may take ~30s)…"):
        try:
            clf = load_classifier()
        except Exception as e:
            st.error(f"❌ Failed to load model:\n\n```\n{e}\n```")
            st.stop()

    st.markdown('<div class="section-label">Upload Audio</div>', unsafe_allow_html=True)

    # ── Upload zone ─────────────────────────
    col_upload, col_info = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            label="Upload an audio file (.wav or .mp3, max 10 MB)",
            type=["wav", "mp3"],
            help="Drag and drop or click to browse",
            label_visibility="collapsed",
        )

    with col_info:
        st.markdown("""
        <div class="glass-card" style="padding:1rem; margin-top:0;">
            <div style="font-size:0.75rem; color:#8b949e; margin-bottom:0.5rem; font-weight:600; letter-spacing:1px; text-transform:uppercase;">Supported</div>
            <div style="font-size:0.9rem; color:#e6edf3;">📁 .wav, .mp3</div>
            <div style="font-size:0.9rem; color:#e6edf3; margin-top:0.3rem;">📏 Max 10 MB</div>
            <div style="font-size:0.9rem; color:#e6edf3; margin-top:0.3rem;">⏱ Min 1 second</div>
            <div style="font-size:0.75rem; color:#8b949e; margin-top:0.75rem;">50 ESC-50 classes</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Once file is uploaded ───────────────
    if uploaded_file is not None:

        # File size validation
        file_bytes = uploaded_file.getvalue()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.")
            st.stop()

        st.markdown("---")
        col_player, col_btn = st.columns([3, 1])

        with col_player:
            st.markdown('<div class="section-label">Audio Preview</div>', unsafe_allow_html=True)
            st.audio(file_bytes, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            st.caption(f"📄 `{uploaded_file.name}` — {size_mb:.2f} MB")

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            classify_clicked = st.button("🔍 Classify Audio", key="classify_btn")

        # ── Classification ──────────────────
        if classify_clicked:
            # Save uploaded file to temp
            tmp_suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
            tmp_path = os.path.join(TEMP_DIR, "upload" + tmp_suffix)
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)

            with st.spinner("🧠 Encoding audio and computing similarity scores…"):
                t0 = time.time()
                try:
                    results = clf.classify(tmp_path, top_n=TOP_N)
                    elapsed = time.time() - t0
                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Classification failed:\n\n```\n{e}\n```")
                    st.stop()
                finally:
                    # Cleanup raw upload
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            # ── Results ────────────────────
            st.markdown("---")
            st.markdown('<div class="section-label">Classification Results</div>', unsafe_allow_html=True)

            top = results[0]
            cat_color = CATEGORY_COLORS.get(top["category"], "#8b949e")

            # Summary card
            st.markdown(f"""
            <div class="result-summary">
                <div class="result-rank-icon">🥇</div>
                <div class="result-label">{top["label"].upper()}</div>
                <div class="result-score">{top["score"]*100:.1f}% similarity</div>
                <span class="category-pill" style="background:{cat_color}22; color:{cat_color}; border:1px solid {cat_color}55;">
                    {top["category"]}
                </span>
                <div style="font-size:0.75rem; color:#6e7681; margin-top:0.75rem;">
                    ⏱ Processed in {elapsed:.2f}s
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Bar chart
            fig = build_bar_chart(results)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Top-10 table
            st.markdown('<div class="section-label">Full Rankings</div>', unsafe_allow_html=True)

            cols = st.columns([0.5, 2.5, 1.5, 2])
            headers = ["Rank", "Label", "Score", "Category"]
            for col, h in zip(cols, headers):
                col.markdown(f"**{h}**")

            st.markdown('<hr style="margin: 0.25rem 0 0.5rem 0;">', unsafe_allow_html=True)

            for r in results:
                cols = st.columns([0.5, 2.5, 1.5, 2])
                medal = RANK_MEDALS.get(r["rank"], f"#{r['rank']}")
                cat_c = CATEGORY_COLORS.get(r["category"], "#8b949e")
                cols[0].write(medal)
                cols[1].write(r["label"].title())
                cols[2].markdown(
                    f'<span style="color:#58a6ff; font-weight:600;">{r["score"]*100:.1f}%</span>',
                    unsafe_allow_html=True
                )
                cols[3].markdown(
                    f'<span style="color:{cat_c}; font-size:0.85rem;">{r["category"]}</span>',
                    unsafe_allow_html=True
                )

    else:
        # placeholder hint
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color:#484f58;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎵</div>
            <div style="font-size:1rem;">Upload an audio file above to begin classification</div>
            <div style="font-size:0.8rem; margin-top:0.5rem;">Supports .wav and .mp3 — up to 10 MB</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ──────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#484f58; font-size:0.78rem; padding: 0.5rem 0 1rem 0;">
        Based on <a href="https://arxiv.org/pdf/2306.12300.pdf" target="_blank" style="color:#58a6ff;">
        A Multimodal Prototypical Approach for Unsupervised Sound Classification</a>
        · INTERSPEECH 2023 ·
        <a href="https://github.com/sakshamsingh1/audio_text_proto" target="_blank" style="color:#58a6ff;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
