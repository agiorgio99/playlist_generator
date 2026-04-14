#!/usr/bin/env python3
"""
app_similarity.py — Streamlit app for track similarity search.

Select a query track and view the top-10 most similar tracks using
Discogs-Effnet and CLAP embeddings side by side.

Run:  streamlit run app_similarity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_PARQUET,
    cosine_similarity_matrix,
    load_features,
    render_track_list,
)

# ──────────────────────────────────────────────────────────────────────
CONFIG = {
    "parquet_path": DEFAULT_PARQUET,
    "audio_dir": DEFAULT_AUDIO_DIR,
}


# ──────────────────────────────────────────────────────────────────────
def main():
    """Main Streamlit application for track similarity."""
    st.set_page_config(page_title="Track Similarity", layout="wide")
    st.title("🔍 Track Similarity Search")

    df = load_features(CONFIG["parquet_path"])

    has_effnet = "effnet_embedding" in df.columns and df["effnet_embedding"].notna().any()
    has_clap = "clap_embedding" in df.columns and df["clap_embedding"].notna().any()

    if not has_effnet and not has_clap:
        st.error("No embedding vectors found in features.parquet. "
                 "Ensure analyse_collection.py ran with Effnet and/or CLAP models.")
        st.stop()

    # Build dropdown options
    track_options = df["track_id"].tolist()
    query_id = st.selectbox("Select a query track", track_options)

    query_row = df[df["track_id"] == query_id].iloc[0]

    st.subheader("Query track")
    audio_path = CONFIG["audio_dir"] / query_row["file"]
    if audio_path.exists():
        st.audio(str(audio_path))
    st.caption(
        f"BPM: {query_row.get('bpm', 'N/A'):.0f} · "
        f"Danceability: {query_row.get('danceability', 'N/A'):.2f}"
        if pd.notna(query_row.get("bpm")) else query_id
    )

    st.divider()

    # ── Compute similarities ─────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Discogs-Effnet
    with col_left:
        st.subheader("Top-10 — Discogs-Effnet")
        if has_effnet:
            effnet_df = df[df["effnet_embedding"].notna()].copy()
            corpus = np.stack(effnet_df["effnet_embedding"].values)
            query_vec = np.array(query_row["effnet_embedding"])

            sims = cosine_similarity_matrix(query_vec, corpus)
            effnet_df = effnet_df.copy()
            effnet_df["sim_effnet"] = sims

            # Exclude query track itself
            effnet_df = effnet_df[effnet_df["track_id"] != query_id]
            effnet_df = effnet_df.sort_values("sim_effnet", ascending=False)

            render_track_list(
                effnet_df,
                n=10,
                audio_dir=CONFIG["audio_dir"],
                show_similarity=True,
                sim_col="sim_effnet",
            )
        else:
            st.info("Effnet embeddings not available.")

    # CLAP
    with col_right:
        st.subheader("Top-10 — CLAP")
        if has_clap:
            clap_df = df[df["clap_embedding"].notna()].copy()
            corpus = np.stack(clap_df["clap_embedding"].values)
            query_vec = np.array(query_row["clap_embedding"])

            sims = cosine_similarity_matrix(query_vec, corpus)
            clap_df = clap_df.copy()
            clap_df["sim_clap"] = sims

            clap_df = clap_df[clap_df["track_id"] != query_id]
            clap_df = clap_df.sort_values("sim_clap", ascending=False)

            render_track_list(
                clap_df,
                n=10,
                audio_dir=CONFIG["audio_dir"],
                show_similarity=True,
                sim_col="sim_clap",
            )
        else:
            st.info("CLAP embeddings not available.")


if __name__ == "__main__":
    main()
