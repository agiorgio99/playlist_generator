#\!/usr/bin/env python3
"""
app_text_query.py - Streamlit app for freeform text search via CLAP.

Run:  streamlit run app_text_query.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    DEFAULT_CLAP_CHECKPOINT,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_BASE,
    collection_picker_ui,
    cosine_similarity_matrix,
    export_m3u8,
    load_features,
    render_track_list,
)

EXAMPLE_QUERIES = [
    "upbeat electronic dance music with synthesizers",
    "calm acoustic guitar folk music",
    "heavy metal with distorted guitars and drums",
    "jazz with piano and saxophone",
    "sad slow piano ballad",
    "energetic hip hop with strong bass",
    "relaxing ambient instrumental music",
    "reggae with tropical rhythm",
]


@st.cache_resource
def load_clap_model(models_dir: str, checkpoint: str):
    ckpt_path = Path(models_dir) / checkpoint
    if not ckpt_path.exists():
        st.error(
            f"CLAP checkpoint not found at {ckpt_path}. "
            "Run analyze_collection.py — it downloads the model automatically."
        )
        return None
    try:
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device="cpu")
        model.load_ckpt(str(ckpt_path))
        return model
    except Exception as exc:
        st.error(f"Failed to load CLAP model: {exc}")
        return None


@st.cache_data
def build_clap_corpus(parquet_path: str):
    df      = pd.read_parquet(parquet_path)
    clap_df = df[df["clap_embedding"].notna()].copy().reset_index(drop=True)
    if clap_df.empty:
        return clap_df, np.empty((0, 0))
    matrix = np.stack(clap_df["clap_embedding"].values)
    return clap_df, matrix


def encode_text(clap_model, text: str) -> np.ndarray:
    emb = clap_model.get_text_embedding([text], use_tensor=False)
    return np.array(emb).squeeze(0)


def run_search(query: str, clap_model, clap_df, corpus_matrix, n_results, audio_dir):
    """Encode query and display results."""
    with st.spinner("Searching…"):
        text_emb  = encode_text(clap_model, query)
        sims      = cosine_similarity_matrix(text_emb, corpus_matrix)
        result_df = clap_df.copy()
        result_df["sim_text"] = sims
        result_df = result_df.sort_values("sim_text", ascending=False)

    st.subheader(f'Top {n_results} results for: "{query}"')
    render_track_list(
        result_df, n=n_results, audio_dir=audio_dir,
        show_similarity=True, sim_col="sim_text",
    )
    st.divider()
    st.download_button(
        label="Export results (.m3u8)",
        data=export_m3u8(result_df.head(n_results), audio_dir),
        file_name="text_query_playlist.m3u8",
        mime="audio/x-mpegurl",
    )


def main():
    st.set_page_config(page_title="Text to Music Search", layout="wide")
    st.title("Text-to-Music Search (CLAP)")
    st.caption(
        "Describe the music you want in plain language. "
        "CLAP encodes your query and ranks the closest matching tracks."
    )

    # ── Session state ────────────────────────────────────────────────────────
    if "active_query" not in st.session_state:
        st.session_state.active_query = ""

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

    n_results = st.sidebar.slider("Number of results", 5, 20, 10, 5)

    # ── Load data ────────────────────────────────────────────────────────────
    df = load_features(parquet_path)

    if "clap_embedding" not in df.columns or df["clap_embedding"].isna().all():
        st.error(
            "No CLAP embeddings found in features.parquet. "
            "Re-run analyze_collection.py to compute them."
        )
        st.stop()

    clap_df, corpus_matrix = build_clap_corpus(parquet_path)
    n_clap = len(clap_df)

    if n_clap < 10:
        st.warning(
            f"Only {n_clap} tracks have CLAP embeddings — results will be limited. "
            "Re-run analyze_collection.py to add embeddings for all tracks."
        )
    else:
        st.caption(f"{n_clap} tracks available for text search.")

    with st.spinner("Loading CLAP model…"):
        clap_model = load_clap_model(str(DEFAULT_MODELS_DIR), DEFAULT_CLAP_CHECKPOINT)

    if clap_model is None:
        st.stop()

    # ── Example query buttons ────────────────────────────────────────────────
    st.subheader("Describe the music")
    st.markdown("**Try an example** *(click to search immediately)*:")
    cols = st.columns(4)
    for i, example in enumerate(EXAMPLE_QUERIES):
        if cols[i % 4].button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state.active_query = example

    st.markdown("")  # spacer

    # ── Custom text input ────────────────────────────────────────────────────
    custom = st.text_input(
        "Or type your own query and press Enter",
        placeholder="e.g. relaxing jazz piano late at night",
        key="custom_input",
    )
    if custom.strip():
        st.session_state.active_query = custom.strip()

    # ── Search ───────────────────────────────────────────────────────────────
    active = st.session_state.active_query
    if active:
        # Show the active query as a pill so the user knows what's running
        st.info(f"🔍 **Active query:** {active}")
        run_search(active, clap_model, clap_df, corpus_matrix, n_results, audio_dir)


if __name__ == "__main__":
    main()
