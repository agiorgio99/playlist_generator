#!/usr/bin/env python3
"""
app_descriptors.py — Streamlit app for playlist generation by audio descriptors.

Sidebar controls for style, tempo, voice/instrumental, danceability, and key.
Filters and ranks matching tracks, shows top-10 with audio players, and
offers M3U8 export.

Run:  streamlit run app_descriptors.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_PARQUET,
    export_m3u8,
    load_features,
    render_audio_player,
)

# ──────────────────────────────────────────────────────────────────────
CONFIG = {
    "parquet_path": DEFAULT_PARQUET,
    "audio_dir": DEFAULT_AUDIO_DIR,
    "models_dir": Path("./models"),
}

NOTE_NAMES = ["Any", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALE_OPTIONS = ["Any", "major", "minor"]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def load_discogs_labels(models_dir: Path) -> list[str]:
    """Load Discogs-400 label names, trying JSON then TXT fallback."""
    json_path = models_dir / "genre_discogs400-discogs-effnet-1.json"
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        return meta.get("classes", [])
    txt_path = models_dir / "discogs400_labels.txt"
    if txt_path.exists():
        return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    return []


def get_top_styles(df: pd.DataFrame, labels: list[str], n: int = 50) -> list[str]:
    """Return the n most common predicted styles across all tracks."""
    if "genre_discogs400" not in df.columns or df["genre_discogs400"].isna().all():
        return []
    if not labels:
        return []

    genre_vecs = np.stack(df["genre_discogs400"].dropna().values)
    top_idxs = np.argmax(genre_vecs, axis=1)
    style_names = [labels[i] for i in top_idxs if i < len(labels)]
    counts = Counter(style_names)
    return [s for s, _ in counts.most_common(n)]


# ──────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────
def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Playlist by Descriptors", layout="wide")
    st.title("🎵 Playlist Generation by Descriptors")

    df = load_features(CONFIG["parquet_path"])
    labels = load_discogs_labels(CONFIG["models_dir"])

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    # Style multiselect
    top_styles = get_top_styles(df, labels)
    selected_styles = []
    style_threshold = 0.0
    if top_styles:
        selected_styles = st.sidebar.multiselect("Music style", top_styles)
        style_threshold = st.sidebar.slider(
            "Min style activation", 0.0, 1.0, 0.1, 0.01
        )

    # Tempo
    bpm_min_val = float(df["bpm"].min()) if df["bpm"].notna().any() else 40.0
    bpm_max_val = float(df["bpm"].max()) if df["bpm"].notna().any() else 220.0
    bpm_range = st.sidebar.slider(
        "Tempo (BPM)",
        min_value=40.0,
        max_value=260.0,
        value=(bpm_min_val, bpm_max_val),
        step=1.0,
    )

    # Voice / Instrumental
    voice_filter = st.sidebar.radio("Voice / Instrumental", ["All", "Vocal", "Instrumental"])

    # Danceability
    dance_range = st.sidebar.slider("Danceability", 0.0, 1.0, (0.0, 1.0), 0.01)

    # Key
    key_filter = st.sidebar.selectbox("Key", NOTE_NAMES)
    scale_filter = st.sidebar.selectbox("Scale", SCALE_OPTIONS)

    # ── Filtering ────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)

    # BPM
    if df["bpm"].notna().any():
        mask &= df["bpm"].between(bpm_range[0], bpm_range[1]) | df["bpm"].isna()

    # Voice / Instrumental
    if voice_filter == "Vocal" and "voice_prob" in df.columns:
        mask &= df["voice_prob"] >= 0.5
    elif voice_filter == "Instrumental" and "voice_prob" in df.columns:
        mask &= df["voice_prob"] < 0.5

    # Danceability
    if "danceability" in df.columns and df["danceability"].notna().any():
        mask &= df["danceability"].between(dance_range[0], dance_range[1]) | df["danceability"].isna()

    # Key (use temperley profile by default)
    if key_filter != "Any" and "key_temperley" in df.columns:
        mask &= df["key_temperley"] == key_filter
    if scale_filter != "Any" and "scale_temperley" in df.columns:
        mask &= df["scale_temperley"] == scale_filter

    # Style
    if selected_styles and labels and "genre_discogs400" in df.columns:
        style_indices = [labels.index(s) for s in selected_styles if s in labels]
        if style_indices:

            def style_match(vec):
                """Check if any selected style activation exceeds threshold."""
                if vec is None or (isinstance(vec, float) and np.isnan(vec)):
                    return False
                arr = np.array(vec)
                return any(arr[i] >= style_threshold for i in style_indices if i < len(arr))

            style_mask = df["genre_discogs400"].apply(style_match)
            mask &= style_mask

    filtered = df[mask].copy()

    # ── Results ──────────────────────────────────────────────────────
    st.subheader(f"Matching tracks: {len(filtered)}")

    if filtered.empty:
        st.info("No tracks match the current filters. Try relaxing your criteria.")
        return

    # Sort by danceability (descending) as a default ranking
    if "danceability" in filtered.columns and filtered["danceability"].notna().any():
        filtered = filtered.sort_values("danceability", ascending=False)

    # Show top 10
    for _, row in filtered.head(10).iterrows():
        render_audio_player(row, CONFIG["audio_dir"])
        st.divider()

    # ── Export ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Export playlist (.m3u8)"):
        m3u8_content = export_m3u8(filtered, CONFIG["audio_dir"])
        st.sidebar.download_button(
            label="Download .m3u8",
            data=m3u8_content,
            file_name="playlist.m3u8",
            mime="audio/x-mpegurl",
        )


if __name__ == "__main__":
    main()
