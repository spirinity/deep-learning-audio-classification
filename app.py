"""
Streamlit web app for ESC-50 sound classification method comparison.
"""

import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Sound Classifier Comparison",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data", "input", "630k-audioset-fusion-best.pt")
PROTOTYPE_PATH = os.path.join(BASE_DIR, "data", "demo", "mean_embd_tensor_esc50_clap_zs.pt")
LABEL_CSV_PATH = os.path.join(BASE_DIR, "data", "labels", "esc50.csv")
LOGREG_PATH = os.path.join(BASE_DIR, "data", "demo", "logreg_esc50_clap.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "data", "demo", "comparison_metrics.csv")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

MAX_FILE_SIZE_MB = 10
TOP_N = 10

os.makedirs(TEMP_DIR, exist_ok=True)


st.markdown("""
<style>
html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 52%, #0d1117 100%);
    color: #e6edf3;
}
.main-header { text-align:center; padding:2.2rem 0 1.2rem 0; }
.main-title {
    font-size:2.8rem; font-weight:800; margin-bottom:0.35rem;
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 52%, #56d364 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.main-subtitle { color:#8b949e; font-size:1rem; }
.paper-badge {
    display:inline-block; margin-top:0.7rem; padding:0.25rem 0.75rem;
    color:#58a6ff; border:1px solid rgba(88,166,255,0.35);
    background:rgba(88,166,255,0.1); border-radius:18px; font-size:0.75rem; font-weight:600;
}
.glass-card {
    background: rgba(22, 27, 34, 0.82);
    border: 1px solid rgba(48, 54, 61, 0.85);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}
.result-summary {
    background: linear-gradient(135deg, rgba(88,166,255,0.08), rgba(163,113,247,0.08));
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 12px;
    padding: 1.15rem;
    text-align: center;
    min-height: 155px;
}
.section-label {
    font-size:0.72rem; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; color:#8b949e; margin-bottom:0.55rem;
}
.result-label { font-size:1.35rem; font-weight:800; color:#e6edf3; margin-top:0.3rem; }
.result-score { font-size:1.15rem; color:#58a6ff; font-weight:700; }
.category-pill {
    display:inline-block; padding:0.28rem 0.75rem; border-radius:18px;
    font-size:0.78rem; font-weight:700; margin-top:0.45rem;
}
.stAlert { border-radius:10px !important; }
hr { border-color: rgba(48,54,61,0.65) !important; }
</style>
""", unsafe_allow_html=True)


CATEGORY_COLORS = {
    "Animals": "#f78166",
    "Natural soundscapes": "#56d364",
    "Human non-speech": "#ffa657",
    "Interior/domestic": "#58a6ff",
    "Exterior/urban": "#a371f7",
    "Unknown": "#8b949e",
}

RANK_MEDALS = {1: "1", 2: "2", 3: "3"}


@st.cache_resource(show_spinner=False)
def load_classifier():
    from classifier import SoundClassifier

    clf = SoundClassifier(MODEL_PATH, PROTOTYPE_PATH, LABEL_CSV_PATH, LOGREG_PATH)
    clf.load_model()
    clf.load_prototypes()
    clf.load_labels()
    clf.load_text_embeddings()
    clf.load_logistic_model()
    return clf


def check_setup() -> dict:
    return {
        "LAION-CLAP model": (MODEL_PATH, os.path.exists(MODEL_PATH), True),
        "Prototype embeddings": (PROTOTYPE_PATH, os.path.exists(PROTOTYPE_PATH), True),
        "ESC-50 label CSV": (LABEL_CSV_PATH, os.path.exists(LABEL_CSV_PATH), True),
        "Logistic Regression artifact": (LOGREG_PATH, os.path.exists(LOGREG_PATH), False),
    }


def required_setup_ok(status: dict) -> bool:
    return all(exists for _, exists, required in status.values() if required)


def render_setup_status(status: dict):
    if required_setup_ok(status):
        st.success("Required files are ready.")
    else:
        st.error("Some required files are missing.")

    rows = []
    for name, (path, exists, required) in status.items():
        rows.append({
            "item": name,
            "required": "yes" if required else "optional",
            "status": "found" if exists else "missing",
            "path": os.path.relpath(path, BASE_DIR),
        })
    st.dataframe(rows, use_container_width=True)

    if not status["Logistic Regression artifact"][1]:
        st.info("Untuk mengaktifkan Logistic Regression di web app, jalankan `python evaluate_methods.py`, lalu restart Streamlit.")


def build_rank_chart(results: list) -> go.Figure:
    labels = [r["label"].title() for r in results]
    scores = [r["score"] * 100 for r in results]
    fig = go.Figure(go.Bar(
        x=scores[::-1],
        y=labels[::-1],
        orientation="h",
        marker=dict(color="#58a6ff"),
        text=[f"{s:.1f}%" for s in scores[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        xaxis=dict(range=[0, 115], ticksuffix="%", gridcolor="rgba(48,54,61,0.55)"),
        yaxis=dict(color="#e6edf3"),
        margin=dict(l=10, r=60, t=15, b=20),
        height=410,
    )
    return fig


def build_method_chart(method_results: dict) -> go.Figure:
    methods, labels, scores = [], [], []
    for method, results in method_results.items():
        top = results[0]
        methods.append(method)
        labels.append(top["label"].title())
        scores.append(top["score"] * 100)

    fig = go.Figure(go.Bar(
        x=methods,
        y=scores,
        marker=dict(color=["#58a6ff", "#a371f7", "#56d364"][:len(methods)]),
        text=[f"{score:.1f}%<br>{label}" for score, label in zip(scores, labels)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        yaxis=dict(range=[0, 115], ticksuffix="%", gridcolor="rgba(48,54,61,0.55)"),
        xaxis=dict(color="#e6edf3"),
        margin=dict(l=10, r=20, t=15, b=20),
        height=360,
    )
    return fig


def render_method_status(clf):
    status = clf.get_method_status()
    cols = st.columns(3)
    for col, (method, info) in zip(cols, status.items()):
        icon = "Ready" if info["available"] else "Missing"
        col.markdown(f"""
        <div class="glass-card" style="min-height:145px;">
            <div style="font-weight:800; color:#e6edf3;">{method}</div>
            <div style="font-size:0.78rem; color:#8b949e; margin:0.25rem 0 0.55rem 0;">{icon}</div>
            <div style="font-size:0.84rem; color:#c9d1d9;">{info["description"]}</div>
            <div style="font-size:0.72rem; color:#8b949e; margin-top:0.75rem;">{info["setup"]}</div>
        </div>
        """, unsafe_allow_html=True)


def render_evaluation_metrics():
    if not os.path.exists(METRICS_PATH):
        st.info("Belum ada hasil evaluasi dataset. Jalankan `python evaluate_methods.py` untuk membuat comparison_metrics.csv.")
        return

    metrics = pd.read_csv(METRICS_PATH)
    if metrics.empty:
        st.warning("comparison_metrics.csv kosong.")
        return

    summary = (
        metrics.groupby("method", as_index=False)
        .agg({
            "accuracy": "mean",
            "macro_f1": "mean",
            "top3_accuracy": "mean",
            "avg_inference_time_sec": "mean",
        })
        .sort_values("accuracy", ascending=False)
    )
    st.dataframe(
        summary.style.format({
            "accuracy": "{:.3f}",
            "macro_f1": "{:.3f}",
            "top3_accuracy": "{:.3f}",
            "avg_inference_time_sec": "{:.4f}",
        }),
        use_container_width=True,
    )


def render_prediction_cards(method_results: dict):
    cols = st.columns(len(method_results))
    for col, (method, results) in zip(cols, method_results.items()):
        top = results[0]
        color = CATEGORY_COLORS.get(top["category"], "#8b949e")
        col.markdown(f"""
        <div class="result-summary">
            <div style="font-size:0.72rem; color:#8b949e; font-weight:800; letter-spacing:1px; text-transform:uppercase;">{method}</div>
            <div class="result-label">{top["label"].upper()}</div>
            <div class="result-score">{top["score"]*100:.1f}%</div>
            <span class="category-pill" style="background:{color}22; color:{color}; border:1px solid {color}55;">
                {top["category"]}
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_rank_table(results: list):
    rows = []
    for item in results:
        rows.append({
            "rank": RANK_MEDALS.get(item["rank"], str(item["rank"])),
            "label": item["label"].title(),
            "score": f"{item['score'] * 100:.1f}%",
            "raw_score": f"{item['raw_score']:.4f}",
            "category": item["category"],
        })
    st.dataframe(rows, use_container_width=True)


def main():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Sound Classifier Comparison</div>
        <div class="main-subtitle">Zero-Shot CLAP vs Proto-LC vs Logistic Regression on ESC-50</div>
        <span class="paper-badge">INTERSPEECH 2023 · LAION-CLAP · ESC-50</span>
    </div>
    """, unsafe_allow_html=True)

    setup_status = check_setup()
    with st.expander("Setup Status", expanded=not required_setup_ok(setup_status)):
        render_setup_status(setup_status)
    if not required_setup_ok(setup_status):
        st.stop()

    with st.spinner("Loading LAION-CLAP model and method artifacts..."):
        try:
            clf = load_classifier()
        except Exception as e:
            st.error(f"Failed to load model:\n\n```\n{e}\n```")
            st.stop()

    with st.expander("Method Explanation", expanded=False):
        render_method_status(clf)

    with st.expander("ESC-50 Evaluation Metrics", expanded=False):
        render_evaluation_metrics()

    st.markdown('<div class="section-label">Upload Audio</div>', unsafe_allow_html=True)
    col_upload, col_info = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            label="Upload audio file",
            type=["wav", "mp3"],
            label_visibility="collapsed",
        )
    with col_info:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.75rem; color:#8b949e; font-weight:800; letter-spacing:1px; text-transform:uppercase;">Supported</div>
            <div style="font-size:0.9rem; color:#e6edf3; margin-top:0.45rem;">.wav, .mp3</div>
            <div style="font-size:0.9rem; color:#e6edf3;">Max 10 MB</div>
            <div style="font-size:0.9rem; color:#e6edf3;">Min 1 second</div>
            <div style="font-size:0.75rem; color:#8b949e; margin-top:0.65rem;">50 ESC-50 classes</div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem 0; color:#484f58;">
            <div style="font-size:1rem;">Upload an audio file above to compare methods.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        file_bytes = uploaded_file.getvalue()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.")
            st.stop()

        st.markdown("---")
        col_player, col_btn = st.columns([3, 1])
        with col_player:
            st.markdown('<div class="section-label">Audio Preview</div>', unsafe_allow_html=True)
            st.audio(file_bytes, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            st.caption(f"`{uploaded_file.name}` - {size_mb:.2f} MB")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            classify_clicked = st.button("Compare Methods", key="classify_btn", use_container_width=True)

        if classify_clicked:
            tmp_suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
            tmp_path = os.path.join(TEMP_DIR, "upload" + tmp_suffix)
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)

            with st.spinner("Encoding audio and comparing methods..."):
                t0 = time.time()
                try:
                    method_results = clf.classify_all(tmp_path, top_n=TOP_N)
                    elapsed = time.time() - t0
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Classification failed:\n\n```\n{e}\n```")
                    st.stop()
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            st.markdown("---")
            st.markdown('<div class="section-label">Method Comparison Results</div>', unsafe_allow_html=True)
            if "Logistic Regression" not in method_results:
                st.info("Logistic Regression belum tersedia. Jalankan `python evaluate_methods.py`, lalu restart Streamlit.")

            render_prediction_cards(method_results)
            st.caption(f"Processed in {elapsed:.2f}s. CLAP audio embedding dihitung sekali, lalu dipakai oleh semua metode.")
            st.plotly_chart(build_method_chart(method_results), use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="section-label">Top Rankings by Method</div>', unsafe_allow_html=True)
            tabs = st.tabs(list(method_results.keys()))
            for tab, (method, results) in zip(tabs, method_results.items()):
                with tab:
                    st.plotly_chart(build_rank_chart(results), use_container_width=True, config={"displayModeBar": False})
                    render_rank_table(results)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#484f58; font-size:0.78rem; padding:0.5rem 0 1rem 0;">
        Based on A Multimodal Prototypical Approach for Unsupervised Sound Classification · INTERSPEECH 2023
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
