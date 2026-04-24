#\!/usr/bin/env python3
"""
app_similarity.py - Streamlit app for track similarity search.

Run:  streamlit run app_similarity.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_BASE,
    collection_picker_ui,
    cosine_similarity_matrix,
    export_m3u8,
    find_audio_file,
    load_features,
    render_track_list,
    top_genre_label,
)


@st.cache_data
def build_effnet_matrix(parquet_path: str):
    df  = pd.read_parquet(parquet_path)
    sub = df[df["effnet_embedding"].notna()].copy().reset_index(drop=True)
    matrix = np.stack(sub["effnet_embedding"].values)
    return sub, matrix


@st.cache_data
def build_clap_matrix(parquet_path: str):
    df  = pd.read_parquet(parquet_path)
    sub = df[df["clap_embedding"].notna()].copy().reset_index(drop=True)
    if sub.empty:
        return sub, np.empty((0, 0))
    matrix = np.stack(sub["clap_embedding"].values)
    return sub, matrix


@st.cache_data
def load_discogs_labels(models_dir: str) -> list:
    json_path = Path(models_dir) / "genre_discogs400-discogs-effnet-1.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f).get("classes", [])
    return []


def main():
    st.set_page_config(page_title="Track Similarity", layout="wide")
    st.title("Track Similarity Search")
    st.caption(
        "Select a query track and find the most similar tracks using "
        "Discogs-Effnet (music-style embeddings) and CLAP (text-audio embeddings)."
    )

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

    n_results = st.sidebar.slider("Results per embedding", 5, 20, 10, 5)

    df     = load_features(parquet_path)
    labels = load_discogs_labels(str(DEFAULT_MODELS_DIR))

    has_effnet = "effnet_embedding" in df.columns and df["effnet_embedding"].notna().any()
    has_clap   = "clap_embedding"   in df.columns and df["clap_embedding"].notna().any()

    if not has_effnet and not has_clap:
        st.error(
            "No embedding vectors found in features.parquet. "
            "Re-run analyze_collection.py with the Essentia and CLAP models present."
        )
        st.stop()

    n_clap = int(df["clap_embedding"].notna().sum()) if has_clap else 0
    if has_clap and n_clap < 10:
        st.warning(
            f"Only {n_clap} tracks have CLAP embeddings. "
            "Re-run analyze_collection.py to compute them for the full collection."
        )

    # ── Query track selector ─────────────────────────────────────────────────
    st.sidebar.header("Query Track")
    search_term = st.sidebar.text_input(
        "Filter track IDs", placeholder="Type part of a track ID..."
    ).strip()
    track_ids    = df["track_id"].tolist()
    filtered_ids = [t for t in track_ids if search_term.lower() in t.lower()] if search_term else track_ids

    if not filtered_ids:
        st.sidebar.warning("No tracks match your filter.")
        st.stop()

    query_id  = st.sidebar.selectbox("Select query track", filtered_ids)
    query_row = df[df["track_id"] == query_id].iloc[0]

    # ── Query track display ──────────────────────────────────────────────────
    st.subheader("Query track")
    meta_parts  = []
    genre_label = top_genre_label(query_row.get("genre_discogs400"), labels)
    if genre_label:
        meta_parts.append(f"Genre: {genre_label}")
    if pd.notna(query_row.get("bpm")):
        meta_parts.append(f"BPM: {query_row['bpm']:.0f}")
    if pd.notna(query_row.get("danceability")):
        meta_parts.append(f"Dance: {query_row['danceability']:.2f}")
    if pd.notna(query_row.get("voice_prob")):
        meta_parts.append("Vocal" if query_row["voice_prob"] >= 0.5 else "Instrumental")
    if pd.notna(query_row.get("key_temperley")) and pd.notna(query_row.get("scale_temperley")):
        meta_parts.append(f"Key: {query_row['key_temperley']} {query_row['scale_temperley']}")
    if pd.notna(query_row.get("loudness_lufs")):
        meta_parts.append(f"{query_row['loudness_lufs']:.1f} LUFS")
    st.caption(" | ".join(meta_parts) if meta_parts else query_id)

    audio_path = find_audio_file(audio_dir, query_row["file"])
    if audio_path.exists():
        st.audio(str(audio_path))
    else:
        st.warning(f"Audio file not found: {audio_path}")

    st.divider()

    # ── Similarity results ───────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    effnet_results = pd.DataFrame()
    with col_left:
        st.subheader("Top results — Discogs-Effnet")
        if has_effnet and query_row.get("effnet_embedding") is not None:
            effnet_df, effnet_matrix = build_effnet_matrix(parquet_path)
            query_vec = np.array(query_row["effnet_embedding"])
            sims      = cosine_similarity_matrix(query_vec, effnet_matrix)
            effnet_df = effnet_df.copy()
            effnet_df["sim_effnet"] = sims
            effnet_df = effnet_df[effnet_df["track_id"] != query_id]
            effnet_df = effnet_df.sort_values("sim_effnet", ascending=False)
            effnet_results = effnet_df
            render_track_list(
                effnet_df, n=n_results, audio_dir=audio_dir,
                show_similarity=True, sim_col="sim_effnet", labels=labels,
            )
        else:
            st.info("Effnet embeddings not available for this track.")

    clap_results = pd.DataFrame()
    with col_right:
        st.subheader("Top results — CLAP")
        if has_clap and query_row.get("clap_embedding") is not None:
            clap_df, clap_matrix = build_clap_matrix(parquet_path)
            query_vec = np.array(query_row["clap_embedding"])
            sims      = cosine_similarity_matrix(query_vec, clap_matrix)
            clap_df   = clap_df.copy()
            clap_df["sim_clap"] = sims
            clap_df = clap_df[clap_df["track_id"] != query_id]
            clap_df = clap_df.sort_values("sim_clap", ascending=False)
            clap_results = clap_df
            render_track_list(
                clap_df, n=n_results, audio_dir=audio_dir,
                show_similarity=True, sim_col="sim_clap", labels=labels,
            )
        else:
            st.info(
                "CLAP embeddings not available for this track. "
                "Re-run analyze_collection.py to compute them."
            )

    # ── Export ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export playlists")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if not effnet_results.empty:
            st.download_button(
                label="Export Effnet results (.m3u8)",
                data=export_m3u8(effnet_results.head(n_results), audio_dir),
                file_name=f"similar_effnet_{query_id}.m3u8",
                mime="audio/x-mpegurl",
            )
    with exp_col2:
        if not clap_results.empty:
            st.download_button(
                label="Export CLAP results (.m3u8)",
                data=export_m3u8(clap_results.head(n_results), audio_dir),
                file_name=f"similar_clap_{query_id}.m3u8",
                mime="audio/x-mpegurl",
            )


if __name__ == "__main__":
    main()
