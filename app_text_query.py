#!/usr/bin/env python3
"""
app_text_query.py — Streamlit app for freeform text search via CLAP.

Encode a text query with CLAP, compute cosine similarity to all track
CLAP embeddings, and display the top-10 results with audio players.

Run:  streamlit run app_text_query.py
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
    "models_dir": Path("./models"),
    "clap_checkpoint": "music_speech_epoch_15_esc_89.25.pt",
}


# ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_clap_model(models_dir: Path, checkpoint: str):
    """Load the CLAP model once and cache across reruns.

    Parameters
    ----------
    models_dir : Path
        Directory containing the CLAP checkpoint.
    checkpoint : str
        Filename of the CLAP checkpoint.

    Returns
    -------
    laion_clap.CLAP_Module or None
    """
    ckpt_path = models_dir / checkpoint
    if not ckpt_path.exists():
        st.error(f"CLAP checkpoint not found at {ckpt_path}. "
                 "Text search requires the CLAP model.")
        return None
    try:
        import laion_clap

        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt(str(ckpt_path))
        return model
    except Exception as exc:
        st.error(f"Failed to load CLAP model: {exc}")
        return None


def encode_text(clap_model, text: str) -> np.ndarray:
    """Encode a text string into a CLAP embedding vector.

    Parameters
    ----------
    clap_model : laion_clap.CLAP_Module
        Loaded CLAP model.
    text : str
        Free-form text query.

    Returns
    -------
    np.ndarray, shape (d,)
    """
    emb = clap_model.get_text_embedding([text], use_tensor=False)
    return emb.squeeze(0)


# ──────────────────────────────────────────────────────────────────────
def main():
    """Main Streamlit application for text-to-music search."""
    st.set_page_config(page_title="Text → Music Search", layout="wide")
    st.title("📝 Text-to-Music Search (CLAP)")

    df = load_features(CONFIG["parquet_path"])

    if "clap_embedding" not in df.columns or df["clap_embedding"].isna().all():
        st.error("No CLAP embeddings found in features.parquet. "
                 "Run analyze_collection.py with CLAP enabled.")
        st.stop()

    clap_df = df[df["clap_embedding"].notna()].copy()
    st.caption(f"{len(clap_df)} tracks with CLAP embeddings available.")

    clap_model = load_clap_model(CONFIG["models_dir"], CONFIG["clap_checkpoint"])

    query_text = st.text_input(
        "Describe the music you're looking for",
        placeholder="e.g. upbeat electronic dance music with synths",
    )

    if query_text and clap_model is not None:
        with st.spinner("Encoding query and searching..."):
            text_emb = encode_text(clap_model, query_text)
            corpus = np.stack(clap_df["clap_embedding"].values)
            sims = cosine_similarity_matrix(text_emb, corpus)

            clap_df = clap_df.copy()
            clap_df["sim_text"] = sims
            clap_df = clap_df.sort_values("sim_text", ascending=False)

        st.subheader("Top-10 results")
        render_track_list(
            clap_df,
            n=10,
            audio_dir=CONFIG["audio_dir"],
            show_similarity=True,
            sim_col="sim_text",
        )
    elif query_text and clap_model is None:
        st.warning("CLAP model not loaded — cannot encode text query.")


if __name__ == "__main__":
    main()
