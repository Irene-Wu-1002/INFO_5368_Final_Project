import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "spotify_top50_songs_features.csv"
ARTIFACTS_PATH = ROOT / "artifacts"


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def apply_custom_theme():
    # Tell browsers + dark-mode extensions (Dark Reader, Night Eye, etc.) to
    # leave the page alone. The darkreader-lock meta tag is the official
    # opt-out signal Dark Reader respects.
    st.markdown(
        """
        <script>
        (function () {
            var head = document.head || document.getElementsByTagName('head')[0];
            if (!document.querySelector('meta[name="darkreader-lock"]')) {
                var m1 = document.createElement('meta');
                m1.setAttribute('name', 'darkreader-lock');
                head.appendChild(m1);
            }
            if (!document.querySelector('meta[name="color-scheme"]')) {
                var m2 = document.createElement('meta');
                m2.setAttribute('name', 'color-scheme');
                m2.setAttribute('content', 'only light');
                head.appendChild(m2);
            }
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

        :root {
            color-scheme: only light;
            --brand-1: #6366f1;   /* indigo */
            --brand-2: #8b5cf6;   /* violet */
            --brand-3: #ec4899;   /* pink */
            --ink-900: #0f172a;
            --ink-700: #334155;
            --ink-500: #64748b;
            --ink-300: #cbd5e1;
            --ink-100: #e2e8f0;
            --paper: #ffffff;
            --canvas: #f5f7fb;
            --ring: 0 0 0 3px rgba(99,102,241,0.18);
            --shadow-sm: 0 1px 2px rgba(15,23,42,0.04), 0 1px 1px rgba(15,23,42,0.03);
            --shadow-md: 0 6px 18px -8px rgba(15,23,42,0.12), 0 2px 4px rgba(15,23,42,0.04);
            --shadow-lg: 0 20px 40px -20px rgba(79,70,229,0.25), 0 8px 16px -8px rgba(15,23,42,0.08);
        }
        html, body, .stApp { color-scheme: only light; }

        /* ----- Page canvas & typography ----- */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink-900);
            background:
                radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.08), transparent 60%),
                radial-gradient(900px 500px at 110% 10%, rgba(236,72,153,0.06), transparent 55%),
                var(--canvas);
        }
        header[data-testid="stHeader"] { display: none; }
        div[data-testid="stToolbar"] { display: none; }
        #MainMenu { visibility: hidden; }
        .block-container {
            max-width: 1280px;
            padding-top: 1.0rem;
            padding-bottom: 2.5rem;
        }

        /* ----- Page header ----- */
        .page-header {
            position: relative;
            background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(139,92,246,0.08) 55%, rgba(236,72,153,0.08));
            border: 1px solid rgba(99,102,241,0.15);
            border-radius: 20px;
            padding: 22px 26px;
            margin-bottom: 22px;
            box-shadow: var(--shadow-sm);
            overflow: hidden;
        }
        .page-header::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, var(--brand-1), var(--brand-2), var(--brand-3));
        }
        .page-title {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--ink-900);
            margin: 0;
            line-height: 1.15;
        }
        .page-subtitle {
            color: var(--ink-500);
            margin-top: 6px;
            margin-bottom: 0;
            font-size: 0.95rem;
            font-weight: 500;
        }

        /* ----- Cards (used by containers) ----- */
        .dashboard-card {
            background: var(--paper);
            border: 1px solid var(--ink-100);
            border-radius: 18px;
            padding: 20px 22px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 16px;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        .dashboard-card:hover { box-shadow: var(--shadow-md); }
        div[data-testid="stVerticalBlock"]:has(#input-card-anchor),
        div[data-testid="stVerticalBlock"]:has(#result-card-anchor),
        div[data-testid="stVerticalBlock"]:has(#feature-card-anchor) {
            border: none;
            box-shadow: none;
            background: transparent;
            padding: 0;
            margin-bottom: 8px;
        }
        .card-title {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 10px;
            color: var(--ink-900);
            letter-spacing: -0.01em;
        }
        .result-badge {
            display: inline-block;
            padding: 10px 16px;
            border-radius: 12px;
            font-weight: 700;
            color: white;
            letter-spacing: 0.01em;
            background: linear-gradient(90deg, var(--brand-1), var(--brand-2) 55%, var(--brand-3));
            box-shadow: var(--shadow-lg);
        }
        .soft-note {
            color: var(--ink-500);
            font-size: 0.85rem;
            margin-top: 3px;
        }

        /* ----- Feature importance ----- */
        .fi-card { background: transparent; border: none; padding: 6px 0 0 0; margin-top: 6px; }
        .fi-title {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--ink-900);
            margin-bottom: 14px;
        }
        .fi-row { margin-top: 14px; margin-bottom: 14px; }
        .fi-row-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .fi-name { font-size: 1rem; color: var(--ink-900); font-weight: 600; }
        .fi-value { font-size: 1rem; color: var(--ink-500); font-weight: 700; }
        .fi-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: var(--ink-100);
            overflow: hidden;
            box-shadow: inset 0 1px 2px rgba(15,23,42,0.04);
        }
        .fi-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--brand-1) 0%, var(--brand-2) 60%, var(--brand-3) 100%);
            box-shadow: 0 1px 4px rgba(99,102,241,0.35);
        }

        /* ----- Genre popularity ----- */
        .genre-card-title {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--ink-900);
            margin-bottom: 4px;
        }
        .genre-card-subtitle { color: var(--ink-500); font-size: 0.95rem; margin-bottom: 18px; }
        .genre-row { margin: 14px 0 22px 0; }
        .genre-name { font-size: 1.05rem; font-weight: 600; color: var(--ink-900); margin-bottom: 10px; }
        .genre-track { width: 100%; height: 8px; border-radius: 999px; background: var(--ink-100); position: relative; }
        .genre-box {
            position: absolute; top: -8px; height: 24px; border-radius: 8px;
            background: linear-gradient(90deg, var(--brand-1), var(--brand-2));
            border: 2px solid #ffffff;
            box-shadow: 0 4px 12px -2px rgba(99,102,241,0.35);
        }
        .genre-median { position: absolute; top: -8px; width: 3px; height: 24px; background: #ffffff; border-radius: 999px; }

        /* ----- Sidebar ----- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #fafbff 100%);
            border-right: 1px solid var(--ink-100);
        }
        section[data-testid="stSidebar"] .stRadio > div {
            background: var(--canvas);
            border: 1px solid var(--ink-100);
            border-radius: 14px;
            padding: 10px 10px;
        }
        section[data-testid="stSidebar"] .stRadio label {
            padding: 6px 4px;
            font-weight: 500;
        }

        /* ----- Sliders: recolor from default red to brand gradient ----- */
        div[data-baseweb="slider"] > div > div {
            background: var(--ink-100) !important;
        }
        div[data-baseweb="slider"] > div > div > div {
            background: linear-gradient(90deg, var(--brand-1), var(--brand-2)) !important;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background: #ffffff !important;
            border: 2px solid var(--brand-2) !important;
            box-shadow: 0 2px 8px rgba(99,102,241,0.35) !important;
        }
        div[data-baseweb="slider"] [data-testid="stTickBar"] { display: none; }
        /* tooltip / value label above thumb */
        div[data-baseweb="slider"] [role="slider"] + div,
        div[data-baseweb="slider"] span {
            color: var(--brand-2) !important;
            font-weight: 600 !important;
        }

        /* ----- Number inputs ----- */
        .stNumberInput input, .stTextInput input {
            background: var(--paper) !important;
            border: 1px solid var(--ink-100) !important;
            border-radius: 10px !important;
            color: var(--ink-900) !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        .stNumberInput input:focus, .stTextInput input:focus {
            border-color: var(--brand-2) !important;
            box-shadow: var(--ring) !important;
            outline: none !important;
        }
        .stNumberInput button {
            background: var(--canvas) !important;
            border: 1px solid var(--ink-100) !important;
            color: var(--ink-700) !important;
        }
        .stNumberInput button:hover {
            background: #eef2ff !important;
            color: var(--brand-1) !important;
        }

        /* ----- Primary button ----- */
        .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.1rem !important;
            transition: transform 0.1s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--brand-1), var(--brand-2) 60%, var(--brand-3)) !important;
            border: none !important;
            color: #ffffff !important;
            box-shadow: 0 8px 20px -8px rgba(99,102,241,0.55) !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            filter: brightness(1.05);
            box-shadow: 0 12px 26px -8px rgba(139,92,246,0.55) !important;
        }
        .stButton > button[kind="secondary"] {
            background: var(--paper) !important;
            border: 1px solid var(--ink-100) !important;
            color: var(--ink-900) !important;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: var(--brand-2) !important;
            color: var(--brand-2) !important;
        }

        /* ----- Expanders ----- */
        details[data-testid="stExpander"] {
            background: var(--paper);
            border: 1px solid var(--ink-100) !important;
            border-radius: 14px !important;
            box-shadow: var(--shadow-sm);
            margin-bottom: 12px;
        }
        details[data-testid="stExpander"] summary {
            font-weight: 600;
            color: var(--ink-900);
            padding: 12px 16px !important;
        }
        details[data-testid="stExpander"] summary:hover { color: var(--brand-2); }

        /* ----- Metrics ----- */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(236,72,153,0.04));
            border: 1px solid var(--ink-100);
            border-radius: 14px;
            padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--ink-500) !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            font-weight: 800 !important;
            font-size: 2rem !important;
            background: linear-gradient(90deg, var(--brand-1), var(--brand-3));
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        /* ----- Progress bar ----- */
        div[data-testid="stProgress"] > div > div {
            background: var(--ink-100) !important;
            border-radius: 999px !important;
        }
        div[data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, var(--brand-1), var(--brand-2), var(--brand-3)) !important;
            border-radius: 999px !important;
        }

        /* ----- Selectbox ----- */
        div[data-baseweb="select"] > div {
            background: var(--paper) !important;
            border: 1px solid var(--ink-100) !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="select"]:hover > div { border-color: var(--brand-2) !important; }

        /* ----- Alerts ----- */
        div[data-testid="stAlert"] {
            border-radius: 14px !important;
            border: 1px solid var(--ink-100) !important;
        }

        /* ----- Dataframes / tables ----- */
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--ink-100);
        }

        /* ----- Tabs (if used) ----- */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 8px 14px;
        }
        .stTabs [aria-selected="true"] {
            color: var(--brand-2) !important;
            border-bottom: 2px solid var(--brand-2) !important;
        }

        /* ----- Markdown headers ----- */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            letter-spacing: -0.01em;
            color: var(--ink-900);
        }

        /* ----- Scrollbar ----- */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: var(--ink-300);
            border-radius: 999px;
            border: 2px solid var(--canvas);
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--brand-2); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["hit"] = (df["rank"] <= 10).astype(int)
    return df


@st.cache_data
def load_configs():
    with open(ARTIFACTS_PATH / "scaler_config.json", "r", encoding="utf-8") as f:
        scaler_cfg = json.load(f)
    with open(ARTIFACTS_PATH / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return scaler_cfg, metrics


@st.cache_resource
def load_weights():
    with np.load(ARTIFACTS_PATH / "logistic_regression_weights.npz") as lr_npz:
        lr = {"w": lr_npz["w"], "b": lr_npz["b"]}
    with np.load(ARTIFACTS_PATH / "ann_weights.npz") as ann_npz:
        ann = {
            "W1": ann_npz["W1"],
            "b1": ann_npz["b1"],
            "W2": ann_npz["W2"],
            "b2": ann_npz["b2"],
        }
    return lr, ann


def scale_input(x, scaler):
    x_min = np.array(scaler["min"], dtype=float)
    span = np.array(scaler["span"], dtype=float)
    return (x - x_min) / span


def lr_predict_proba(x_scaled, lr_weights):
    w = lr_weights["w"]
    b = lr_weights["b"][0]
    return float(sigmoid(x_scaled @ w + b))


def ann_predict_proba(x_scaled, ann_weights):
    W1 = ann_weights["W1"]
    b1 = ann_weights["b1"]
    W2 = ann_weights["W2"]
    b2 = ann_weights["b2"]
    h = np.maximum(0.0, x_scaled @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    return float(out.reshape(-1)[0])


def render_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
          <p class="page-title">{title}</p>
          <p class="page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_importance_card(metrics, feature_names, top_n=6):
    importance = np.array(metrics.get("logistic_feature_importance", []), dtype=float)
    if importance.size == 0:
        return
    order = np.argsort(-importance)[:top_n]
    idxs = order.tolist()
    values = importance[idxs]
    total = np.sum(values) + 1e-12
    perc = (values / total) * 100.0

    st.markdown('<span id="feature-card-anchor"></span>', unsafe_allow_html=True)
    st.markdown('<div class="fi-card">', unsafe_allow_html=True)
    st.markdown('<div class="fi-title">Feature Importance</div>', unsafe_allow_html=True)
    for i, idx in enumerate(idxs):
        name = feature_names[idx].replace("_", " ").title()
        pct = float(perc[i])
        st.markdown(
            f"""
            <div class="fi-row">
              <div class="fi-row-top">
                <span class="fi-name">{name}</span>
                <span class="fi-value">{pct:.0f}%</span>
              </div>
              <div class="fi-track">
                <div class="fi-fill" style="width:{pct:.1f}%"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _assign_genre_proxy(df):
    # Dataset has no explicit genre column; build a proxy from audio traits.
    genre = np.full(len(df), "Pop", dtype=object)
    edm_mask = (df["tempo"] >= 124) & (df["energy"] >= 0.22)
    hiphop_mask = (df["tempo"] < 110) & (df["zero_crossing_rate"] >= 0.09)
    rock_mask = (df["energy"] >= 0.2) & (df["spectral_centroid"] >= 2300)

    genre[edm_mask.to_numpy()] = "EDM"
    genre[hiphop_mask.to_numpy()] = "Hip-Hop"
    genre[rock_mask.to_numpy() & (~edm_mask.to_numpy())] = "Rock"
    return pd.Series(genre, index=df.index, name="genre_proxy")


def render_popularity_by_genre(filtered_df):
    genre_order = ["Pop", "Rock", "Hip-Hop", "EDM"]
    df = filtered_df.dropna(
        subset=["primary_artist_popularity", "tempo", "energy", "zero_crossing_rate", "spectral_centroid"]
    ).copy()
    if df.empty:
        st.info("Not enough data to compute genre popularity distribution.")
        return

    df["genre_proxy"] = _assign_genre_proxy(df)

    genre_boxes = []
    for g in genre_order:
        vals = df.loc[df["genre_proxy"] == g, "primary_artist_popularity"].to_numpy(dtype=float)
        if vals.size < 5:
            vals = df["primary_artist_popularity"].to_numpy(dtype=float)
        q1 = float(np.percentile(vals, 25))
        med = float(np.percentile(vals, 50))
        q3 = float(np.percentile(vals, 75))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        left = max(0.0, min(100.0, q1))
        width = max(2.0, min(100.0 - left, q3 - q1))
        median = max(left, min(left + width, med))
        genre_boxes.append(
            {
                "name": g,
                "left": left,
                "width": width,
                "median": median,
                "min": max(0.0, min(100.0, min_v)),
                "max": max(0.0, min(100.0, max_v)),
            }
        )

    st.markdown('<div class="genre-card-title">Popularity by Genre</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="genre-card-subtitle">Distribution of artist popularity across genres (data-driven genre proxy)</div>',
        unsafe_allow_html=True,
    )
    for row in genre_boxes:
        st.markdown(
            f"""
            <div class="genre-row">
              <div class="genre-name">{row['name']}</div>
              <div class="genre-track">
                <div style="position:absolute; left:{row['min']}%; top:2px; width:{max(0.8, row['max'] - row['min'])}%; height:4px; background:#d1d5db; border-radius:999px;"></div>
                <div class="genre-box" style="left:{row['left']}%; width:{row['width']}%;"></div>
                <div class="genre-median" style="left:{row['median']}%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_hit_predictor(df, scaler_cfg, metrics, lr_weights, ann_weights):
    render_page_header("Hit Predictor", "Predict whether a song will reach the top 10")

    feature_names = scaler_cfg["feature_names"]
    scaler = scaler_cfg["scaler"]
    best_model = metrics["best_model"]
    decision_threshold = float(metrics.get("decision_thresholds", {}).get(best_model, 0.5))

    defaults = df[feature_names].median(numeric_only=True).to_dict()
    inputs = {}
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        with st.container():
            st.markdown('<span id="input-card-anchor"></span>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Input Features</div>', unsafe_allow_html=True)
            with st.expander("Audio Features", expanded=True):
                for feat in [
                    "tempo",
                    "energy",
                    "zero_crossing_rate",
                    "spectral_centroid",
                    "spectral_rolloff",
                    "mfcc_1",
                    "mfcc_2",
                    "chroma_mean",
                    "chroma_std",
                ]:
                    default = float(defaults.get(feat, 0.0))
                    if feat == "energy":
                        inputs[feat] = st.slider("Energy", min_value=0.0, max_value=1.0, value=float(np.clip(default, 0, 1)))
                    elif feat == "tempo":
                        inputs[feat] = st.slider("Tempo", min_value=40.0, max_value=220.0, value=float(np.clip(default, 40, 220)))
                    else:
                        inputs[feat] = st.number_input(feat.replace("_", " ").title(), value=default, format="%.6f")
            with st.expander("Artist Features", expanded=True):
                for feat in ["primary_artist_popularity", "max_artist_popularity", "monthly_listeners"]:
                    default = float(defaults.get(feat, 0.0))
                    if "popularity" in feat:
                        inputs[feat] = st.slider(
                            feat.replace("_", " ").title(), min_value=0.0, max_value=100.0, value=float(np.clip(default, 0, 100))
                        )
                    else:
                        inputs[feat] = st.number_input("Monthly Listeners", value=default, format="%.2f")
            with st.expander("Chart Context (Optional)", expanded=False):
                st.number_input("Peak Rank", value=15.0, format="%.1f", disabled=True)
                st.number_input("Weeks on Chart", value=8.0, format="%.1f", disabled=True)
                st.markdown('<div class="soft-note">Proposal keeps these as optional context features.</div>', unsafe_allow_html=True)
            predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        x = np.array([inputs[f] for f in feature_names], dtype=float)
        x_scaled = scale_input(x, scaler)
        lr_prob = lr_predict_proba(x_scaled, lr_weights)
        ann_prob = ann_predict_proba(x_scaled, ann_weights)
        final_prob = ann_prob if best_model == "ann" else lr_prob

        with right:
            with st.container():
                st.markdown('<span id="result-card-anchor"></span>', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Prediction Results</div>', unsafe_allow_html=True)
                st.metric("Hit Probability", f"{final_prob * 100:.2f}%")
                st.progress(float(np.clip(final_prob, 0.0, 1.0)))
                label = "Top 10 Hit" if final_prob >= decision_threshold else "Not Top 10"
                st.markdown(f'<span class="result-badge">{label}</span>', unsafe_allow_html=True)
                st.write("")
                st.write(f"Model used: `{best_model}`")
                st.write(f"Threshold: `{decision_threshold:.3f}`")
                st.write(f"Logistic: `{lr_prob:.4f}` | ANN: `{ann_prob:.4f}`")
            render_feature_importance_card(metrics, feature_names, top_n=6)

    else:
        with right:
            with st.container():
                st.markdown('<span id="result-card-anchor"></span>', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Prediction Results</div>', unsafe_allow_html=True)
                st.info("Enter features and click Predict.")
            render_feature_importance_card(metrics, feature_names, top_n=6)


def page_data_explorer(df):
    render_page_header("Data Explorer", "Explore patterns and relationships in music data")

    years = sorted(pd.to_datetime(df["date"]).dt.year.dropna().unique().tolist())
    artists = sorted(df["primary_artist"].dropna().astype(str).unique().tolist())

    c1, c2, c3 = st.columns([1, 1, 0.8])
    year = c1.selectbox("Year", options=["All"] + years)
    artist = c2.selectbox("Artist", options=["All"] + artists[:200])
    c3.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
    if c3.button("Reset Filters", use_container_width=True):
        year = "All"
        artist = "All"

    filtered = df.copy()
    if year != "All":
        filtered = filtered[pd.to_datetime(filtered["date"]).dt.year == year]
    if artist != "All":
        filtered = filtered[filtered["primary_artist"] == artist]

    st.markdown(f"**Filtered rows:** `{len(filtered)}`")

    chart_layout = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
        title_font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=16, color="#0f172a"),
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0)", bordercolor="#e2e8f0", borderwidth=0),
    )

    col1, col2 = st.columns(2, gap="large")
    fig_hist = px.histogram(
        filtered,
        x="tempo",
        color=filtered["hit"].map({0: "Non-hit", 1: "Hit"}),
        nbins=30,
        barmode="overlay",
        title="Tempo Distribution",
        color_discrete_sequence=["#cbd5e1", "#6366f1"],
    )
    fig_hist.update_layout(**chart_layout)
    with col1:
        st.plotly_chart(fig_hist, use_container_width=True)

    fig_scatter = px.scatter(
        filtered,
        x="energy",
        y="streams",
        color=filtered["hit"].map({0: "Non-hit", 1: "Hit"}),
        title="Energy vs Streams",
        hover_data=["track_name", "primary_artist"],
        color_discrete_sequence=["#cbd5e1", "#ec4899"],
    )
    fig_scatter.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0)))
    fig_scatter.update_layout(**chart_layout)
    with col2:
        st.plotly_chart(fig_scatter, use_container_width=True)

    numeric_cols = [
        "tempo",
        "energy",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_rolloff",
        "mfcc_1",
        "mfcc_2",
        "chroma_mean",
        "chroma_std",
        "primary_artist_popularity",
        "monthly_listeners",
        "streams",
        "rank",
    ]
    corr = filtered[numeric_cols].corr(numeric_only=True)
    heat = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[
                [0.0, "#ec4899"],
                [0.5, "#f8fafc"],
                [1.0, "#6366f1"],
            ],
            zmid=0,
            showscale=True,
        )
    )
    heat.update_layout(title="Correlation Heatmap", **chart_layout)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.plotly_chart(heat, use_container_width=True)

    with col4:
        render_popularity_by_genre(filtered)


def page_model_comparison(metrics):
    render_page_header("Model Comparison", "Performance comparison between prediction models")

    rows = []
    for metric_name in ["accuracy", "f1", "auc"]:
        rows.append(
            {
                "Metric": metric_name.upper(),
                "Logistic Regression": metrics["logistic_regression"][metric_name],
                "ANN": metrics["ann"][metric_name],
            }
        )
    table_df = pd.DataFrame(rows)
    st.markdown("#### Model Performance")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    fpr_lr = np.array(metrics["logistic_regression"]["fpr"])
    tpr_lr = np.array(metrics["logistic_regression"]["tpr"])
    fpr_ann = np.array(metrics["ann"]["fpr"])
    tpr_ann = np.array(metrics["ann"]["tpr"])

    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(x=fpr_lr, y=tpr_lr, mode="lines", name="Logistic Regression",
                   line=dict(color="#6366f1", width=3))
    )
    roc_fig.add_trace(
        go.Scatter(x=fpr_ann, y=tpr_ann, mode="lines", name="ANN",
                   line=dict(color="#ec4899", width=3))
    )
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                   line=dict(color="#cbd5e1", width=2, dash="dash"))
    )
    roc_fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
        title_font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=16, color="#0f172a"),
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0)", bordercolor="#e2e8f0", borderwidth=0),
    )

    st.plotly_chart(roc_fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Logistic Regression**")
        st.caption("Baseline model with interpretable coefficients.")
    with c2:
        st.markdown("**Neural Network (ANN)**")
        st.caption("Captures non-linear interactions; expected stronger AUC.")

    st.write(f"Composite score (LR): `{metrics['composite']['logistic_regression']:.4f}`")
    st.write(f"Composite score (ANN): `{metrics['composite']['ann']:.4f}`")
    st.success(f"Best deployed model: {metrics['best_model']}")
    st.caption(
        f"Class weights: pos={metrics.get('class_weights', {}).get('pos_weight', 1.0):.2f}, "
        f"neg={metrics.get('class_weights', {}).get('neg_weight', 1.0):.2f}"
    )

    st.markdown("### Grid Search (Validation)")
    gs_lr = metrics["grid_search"]["logistic_regression"]["best_params"]
    gs_ann = metrics["grid_search"]["ann"]["best_params"]
    c1, c2 = st.columns(2)
    c1.json({"Logistic Regression best params": gs_lr})
    c2.json({"ANN best params": gs_ann})

    st.markdown("### Temporal Split Evaluation")
    ts = metrics["temporal_split"]
    st.caption(f"Train: before `{ts['split_date']}` | Test: on/after `{ts['split_date']}`")
    if ts["logistic_regression"] and ts["ann"]:
        ts_rows = []
        for metric_name in ["accuracy", "f1", "auc"]:
            ts_rows.append(
                {
                    "Metric": metric_name.upper(),
                    "Logistic Regression (Temporal)": ts["logistic_regression"][metric_name],
                    "ANN (Temporal)": ts["ann"][metric_name],
                }
            )
        st.dataframe(pd.DataFrame(ts_rows), use_container_width=True)
        st.caption(
            f"Thresholds - LR: {ts['logistic_regression'].get('threshold', 0.5):.3f}, "
            f"ANN: {ts['ann'].get('threshold', 0.5):.3f}"
        )
    else:
        st.info("Temporal split currently has insufficient rows for one side of the split.")


def main():
    st.set_page_config(page_title="Song Intelligence Dashboard", layout="wide")
    apply_custom_theme()
    st.sidebar.markdown(
        """
        <div style="padding: 6px 4px 14px 4px;">
          <div style="
            font-family: 'Plus Jakarta Sans','Inter',sans-serif;
            font-weight: 800;
            font-size: 1.35rem;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #6366f1, #8b5cf6 55%, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
          ">Song Intelligence</div>
          <div style="color:#64748b; font-size:0.85rem; margin-top:2px;">
            Dashboard Navigation
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    scaler_cfg, metrics = load_configs()
    lr_weights, ann_weights = load_weights()

    page = st.sidebar.radio(
        "Navigate",
        ["Song Hit Predictor", "Data Explorer Dashboard", "Model Comparison"],
    )

    if page == "Song Hit Predictor":
        page_hit_predictor(df, scaler_cfg, metrics, lr_weights, ann_weights)
    elif page == "Data Explorer Dashboard":
        page_data_explorer(df)
    else:
        page_model_comparison(metrics)


if __name__ == "__main__":
    main()

