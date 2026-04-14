"""
utils.py — Shared utilities for Streamlit apps.

Provides helpers for loading features, computing similarity,
exporting playlists, and rendering audio players.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_PARQUET = Path("./analysis_results/features.parquet")
DEFAULT_AUDIO_DIR = Path("./audio_chunks")


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_features(parquet_path: Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Load features.parquet with caching.

    Parameters
    ----------
    parquet_path : Path
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
    """
    if not parquet_path.exists():
        st.error(f"Features file not found: {parquet_path}. Run analyze_collection.py first.")
        st.stop()
    return pd.read_parquet(parquet_path)


# ──────────────────────────────────────────────────────────────────────
# Similarity
# ──────────────────────────────────────────────────────────────────────
def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a single query and a corpus.

    Parameters
    ----------
    query : np.ndarray, shape (d,)
        The query embedding vector.
    corpus : np.ndarray, shape (n, d)
        Matrix of embedding vectors to compare against.

    Returns
    -------
    np.ndarray, shape (n,)
        Cosine similarity scores.
    """
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10
    corpus_norm = corpus / norms
    return corpus_norm @ query_norm


# ──────────────────────────────────────────────────────────────────────
# Playlist export
# ──────────────────────────────────────────────────────────────────────
def export_m3u8(tracks: pd.DataFrame, audio_dir: Path = DEFAULT_AUDIO_DIR) -> str:
    """Generate an M3U8 playlist string for the given tracks.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame with a 'file' column containing filenames.
    audio_dir : Path
        Base directory where audio files live.

    Returns
    -------
    str
        M3U8-formatted playlist content.
    """
    lines = ["#EXTM3U"]
    for _, row in tracks.iterrows():
        filepath = audio_dir / row["file"]
        lines.append(f"#EXTINF:-1,{row.get('track_id', row['file'])}")
        lines.append(str(filepath))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Audio player
# ──────────────────────────────────────────────────────────────────────
def render_audio_player(
    track_row: pd.Series,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    show_info: bool = True,
):
    """Render an audio player for a single track with optional metadata.

    Parameters
    ----------
    track_row : pd.Series
        A row from the features DataFrame.
    audio_dir : Path
        Directory containing the audio files.
    show_info : bool
        Whether to display track metadata alongside the player.
    """
    filepath = audio_dir / track_row["file"]

    if show_info:
        label_parts = [f"**{track_row.get('track_id', track_row['file'])}**"]
        if pd.notna(track_row.get("bpm")):
            label_parts.append(f"BPM: {track_row['bpm']:.0f}")
        if pd.notna(track_row.get("danceability")):
            label_parts.append(f"Dance: {track_row['danceability']:.2f}")
        if pd.notna(track_row.get("voice_prob")):
            vtype = "Vocal" if track_row["voice_prob"] >= 0.5 else "Instr."
            label_parts.append(vtype)
        st.markdown(" · ".join(label_parts))

    if filepath.exists():
        st.audio(str(filepath))
    else:
        st.warning(f"Audio file not found: {filepath}")


def render_track_list(
    df: pd.DataFrame,
    n: int = 10,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    show_similarity: bool = False,
    sim_col: str = "",
):
    """Render a list of tracks with audio players.

    Parameters
    ----------
    df : pd.DataFrame
        Tracks to display (already sorted by relevance).
    n : int
        Maximum number of tracks to show.
    audio_dir : Path
        Audio directory path.
    show_similarity : bool
        If True, display the similarity score.
    sim_col : str
        Column name holding similarity scores.
    """
    for i, (_, row) in enumerate(df.head(n).iterrows()):
        prefix = ""
        if show_similarity and sim_col and sim_col in row.index:
            prefix = f"Sim: {row[sim_col]:.4f} · "
        label_parts = [f"{prefix}**{row.get('track_id', row['file'])}**"]
        if pd.notna(row.get("bpm")):
            label_parts.append(f"BPM: {row['bpm']:.0f}")
        st.markdown(" · ".join(label_parts))
        filepath = audio_dir / row["file"]
        if filepath.exists():
            st.audio(str(filepath))
        else:
            st.caption(f"File not found: {filepath}")
