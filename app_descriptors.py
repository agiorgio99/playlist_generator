#\!/usr/bin/env python3
"""
app_descriptors.py - Streamlit app for playlist generation by audio descriptors.

Run:  streamlit run app_descriptors.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_BASE,
    collection_picker_ui,
    export_m3u8,
    load_features,
    render_audio_player,
    top_genre_label,
)

NOTE_NAMES    = ["Any", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALE_OPTIONS = ["Any", "major", "minor"]
KEY_PROFILES  = ["temperley", "krumhansl", "edma"]


@st.cache_data
def load_discogs_labels(models_dir: str) -> list:
    json_path = Path(models_dir) / "genre_discogs400-discogs-effnet-1.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f).get("classes", [])
    txt_path = Path(models_dir) / "discogs400_labels.txt"
    if txt_path.exists():
        return [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    return []


def get_style_options(df: pd.DataFrame, labels: list, show_all: bool) -> list:
    if "genre_discogs400" not in df.columns or df["genre_discogs400"].isna().all() or not labels:
        return []
    genre_vecs  = np.stack(df["genre_discogs400"].dropna().values)
    top_idxs    = np.argmax(genre_vecs, axis=1)
    style_names = [labels[i] for i in top_idxs if i < len(labels)]
    counts = Counter(style_names)
    if show_all:
        return sorted(labels, key=lambda s: (-counts.get(s, 0), s))
    return [s for s, _ in counts.most_common(50)]


def compute_style_score(vec, style_indices: list) -> float:
    arr = np.array(vec)
    return float(max(arr[i] for i in style_indices if i < len(arr)))


def main():
    st.set_page_config(page_title="Playlist by Descriptors", layout="wide")
    st.title("Playlist Generation by Descriptors")
    st.caption("Filter and rank tracks from the music collection by audio descriptors.")

    # ── Collection picker ────────────────────────────────────────────────────
    st.sidebar.header("Collection")
    audio_dir, parquet_path, collection_name = collection_picker_ui(DEFAULT_RESULTS_BASE)
    st.sidebar.caption(f"📂 {collection_name}")

    # ── How-to instructions ──────────────────────────────────────────────────
    with st.sidebar.expander("ℹ️ How to add a collection", expanded=False):
        st.markdown(
            "**To analyze a new music collection:**\n\n"
            "1. Open a terminal in the project folder\n"
            "2. Run:\n"
            "   ```\n"
            "   python analyze_collection.py\n"
            "   ```\n"
            "3. Enter your HuggingFace token when prompted\n"
            "   *(free — create one at huggingface.co/settings/tokens)*\n"
            "4. Enter the path to your audio folder\n"
            "5. Wait for analysis to finish, then reload this page\n\n"
            "Results are saved per-collection so nothing gets overwritten."
        )

    # ── Sidebar filters ──────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    show_all_styles = st.sidebar.checkbox(
        "Show all 400 genres", value=False,
        help="By default only the 50 most common genres are listed.",
    )

    df     = load_features(parquet_path)
    labels = load_discogs_labels(str(DEFAULT_MODELS_DIR))

    style_options   = get_style_options(df, labels, show_all_styles)
    selected_styles = []
    style_threshold = 0.1
    if style_options:
        selected_styles = st.sidebar.multiselect(
            "Music style (Discogs-400)", style_options,
            help="Tracks are ranked by their max activation among selected styles.",
        )
        style_threshold = st.sidebar.slider("Min style activation", 0.0, 1.0, 0.1, 0.01)
    elif not labels:
        st.sidebar.info("Genre labels not found. Run analyze_collection.py first.")

    bpm_min = float(df["bpm"].min()) if df["bpm"].notna().any() else 40.0
    bpm_max = float(df["bpm"].max()) if df["bpm"].notna().any() else 220.0
    bpm_range = st.sidebar.slider(
        "Tempo (BPM)", 40.0, 260.0,
        (max(40.0, bpm_min), min(260.0, bpm_max)), 1.0,
    )

    voice_filter = st.sidebar.radio(
        "Voice / Instrumental", ["All", "Vocal", "Instrumental"],
        help="Classified as Vocal when voice probability >= 0.5.",
    )

    dance_range = st.sidebar.slider("Danceability", 0.0, 1.0, (0.0, 1.0), 0.01)

    st.sidebar.markdown("**Key / Scale**")
    key_profile  = st.sidebar.selectbox("Key profile", KEY_PROFILES, index=0)
    key_filter   = st.sidebar.selectbox("Key", NOTE_NAMES)
    scale_filter = st.sidebar.selectbox("Scale", SCALE_OPTIONS)

    n_results = st.sidebar.slider("Max tracks shown (with player)", 5, 20, 10, 5)

    # ── Filtering ────────────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)

    if df["bpm"].notna().any():
        mask &= df["bpm"].between(bpm_range[0], bpm_range[1]) | df["bpm"].isna()

    if voice_filter == "Vocal" and "voice_prob" in df.columns:
        mask &= df["voice_prob"] >= 0.5
    elif voice_filter == "Instrumental" and "voice_prob" in df.columns:
        mask &= df["voice_prob"] < 0.5

    if "danceability" in df.columns and df["danceability"].notna().any():
        mask &= df["danceability"].between(dance_range[0], dance_range[1]) | df["danceability"].isna()

    key_col   = f"key_{key_profile}"
    scale_col = f"scale_{key_profile}"
    if key_filter != "Any" and key_col in df.columns:
        mask &= df[key_col] == key_filter
    if scale_filter != "Any" and scale_col in df.columns:
        mask &= df[scale_col] == scale_filter

    style_indices = []
    if selected_styles and labels and "genre_discogs400" in df.columns:
        style_indices = [labels.index(s) for s in selected_styles if s in labels]
        if style_indices:
            def style_match(vec):
                if vec is None or (isinstance(vec, float) and np.isnan(vec)):
                    return False
                arr = np.array(vec)
                return any(arr[i] >= style_threshold for i in style_indices if i < len(arr))
            mask &= df["genre_discogs400"].apply(style_match)

    filtered = df[mask].copy()

    # ── Ranking ──────────────────────────────────────────────────────────────
    if style_indices and "genre_discogs400" in filtered.columns and filtered["genre_discogs400"].notna().any():
        filtered["_style_score"] = filtered["genre_discogs400"].apply(
            lambda v: compute_style_score(v, style_indices)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
            else 0.0
        )
        filtered  = filtered.sort_values("_style_score", ascending=False)
        rank_note = "Ranked by style activation score"
    elif "danceability" in filtered.columns and filtered["danceability"].notna().any():
        filtered  = filtered.sort_values("danceability", ascending=False)
        rank_note = "Ranked by danceability"
    else:
        rank_note = "No ranking applied"

    # ── Results header ────────────────────────────────────────────────────────
    col_count, col_export = st.columns([3, 1])
    with col_count:
        st.subheader(f"Matching tracks: {len(filtered)}")
        st.caption(rank_note)
    with col_export:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.download_button(
            label="Export playlist (.m3u8)",
            data=export_m3u8(filtered, audio_dir),
            file_name="playlist.m3u8",
            mime="audio/x-mpegurl",
        )

    if filtered.empty:
        st.info("No tracks match the current filters. Try relaxing your criteria.")
        return

    st.markdown(f"**Showing top {min(n_results, len(filtered))} tracks:**")
    for _, row in filtered.head(n_results).iterrows():
        if "_style_score" in row and pd.notna(row["_style_score"]):
            st.caption(f"Style score: {row['_style_score']:.3f}")
        genre_label = top_genre_label(row.get("genre_discogs400"), labels)
        render_audio_player(row, audio_dir, genre_label=genre_label)
        st.divider()


if __name__ == "__main__":
    main()
