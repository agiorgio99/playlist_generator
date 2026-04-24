"""
utils.py - Shared utilities for Streamlit apps.

Provides:
  - Config loading (config.yaml)
  - Collection detection (scans analysis_results/ for analyzed collections)
  - Fast audio file lookup via an in-memory index
  - Feature loading with Streamlit cache
  - Cosine similarity computation
  - M3U8 playlist export
  - Audio player / track-list rendering with genre labels
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ── Config loading ────────────────────────────────────────────────────────────
def load_config(config_path: Path = Path("config.yaml")) -> dict:
    defaults = {
        "audio_dir":    "./audio_chunks",
        "results_base": "./analysis_results",
        "models_dir":   "./models",
        "clap_checkpoint": "music_speech_epoch_15_esc_89.25.pt",
    }
    cfg_path = Path(config_path)
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as f:
                loaded = yaml.safe_load(f) or {}
            defaults.update({k: v for k, v in loaded.items() if v is not None})
        except Exception as exc:
            print(f"Warning: could not read {cfg_path}: {exc}")
    return {
        k: Path(v) if str(k).endswith("_dir") or str(k) == "results_base" else v
        for k, v in defaults.items()
    }


_cfg = load_config()

DEFAULT_AUDIO_DIR:       Path = _cfg["audio_dir"]
DEFAULT_RESULTS_BASE:    Path = _cfg["results_base"]
DEFAULT_MODELS_DIR:      Path = _cfg["models_dir"]
DEFAULT_CLAP_CHECKPOINT: str  = _cfg["clap_checkpoint"]
# Legacy alias (single flat parquet — kept for backward compat)
DEFAULT_PARQUET: Path = DEFAULT_RESULTS_BASE / "features.parquet"


# ── Collection detection ──────────────────────────────────────────────────────
def get_analyzed_collections(results_base: Path = DEFAULT_RESULTS_BASE) -> list:
    """
    Return a list of dicts for every analyzed collection found under results_base.

    A collection is valid if its subfolder contains a non-empty features.parquet.
    Each dict has keys:
        name        - folder name (str)
        parquet     - Path to features.parquet
        audio_dir   - Path to audio files (from collection_info.json, or best-guess)
        info        - full collection_info dict (may be empty if file is absent)
    """
    results_base = Path(results_base)
    if not results_base.exists():
        return []

    collections = []
    for d in sorted(results_base.iterdir()):
        if not d.is_dir():
            continue
        parquet = d / "features.parquet"
        if not parquet.exists() or parquet.stat().st_size == 0:
            continue

        info = {}
        info_file = d / "collection_info.json"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
            except Exception:
                pass

        audio_dir = Path(info.get("audio_dir", "")) if info.get("audio_dir") else DEFAULT_AUDIO_DIR

        collections.append({
            "name":      d.name,
            "parquet":   parquet,
            "audio_dir": audio_dir,
            "info":      info,
        })

    return collections


def collection_picker_ui(results_base: Path = DEFAULT_RESULTS_BASE):
    """
    Streamlit widget: shows a selectbox for analyzed collections.
    Returns (audio_dir, parquet_path_str) for the selected collection,
    or shows a tutorial and calls st.stop() if none are found.
    """
    collections = get_analyzed_collections(results_base)

    if not collections:
        st.error("No analyzed collections found.")
        st.info(
            "**How to analyze your music collection:**\n\n"
            "1. Open a terminal and navigate to this project folder.\n"
            "2. Run:\n"
            "   ```\n"
            "   python analyze_collection.py\n"
            "   ```\n"
            "3. Follow the prompts: enter your HuggingFace token and the path to your audio folder.\n"
            "4. Wait for analysis to complete, then come back here.\n\n"
            "The HuggingFace token is free — create one at https://huggingface.co/settings/tokens"
        )
        st.stop()

    names = [c["name"] for c in collections]
    selected_name = st.sidebar.selectbox(
        "Collection",
        names,
        help="Choose which analyzed collection to explore.",
    )
    coll = next(c for c in collections if c["name"] == selected_name)
    return coll["audio_dir"], str(coll["parquet"]), coll["name"]


# ── Audio file index  (built once per session, O(1) lookup afterwards) ────────
_audio_index: dict = {}
_audio_index_dir: Optional[Path] = None

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}


def _ensure_audio_index(audio_dir: Path) -> None:
    global _audio_index, _audio_index_dir
    audio_dir = Path(audio_dir)
    if _audio_index_dir == audio_dir and _audio_index:
        return
    _audio_index = {}
    for ext in AUDIO_EXTENSIONS:
        for p in audio_dir.rglob(f"*{ext}"):
            _audio_index[p.name] = p
        for p in audio_dir.rglob(f"*{ext.upper()}"):
            _audio_index[p.name] = p
    _audio_index_dir = audio_dir


def find_audio_file(audio_dir: Path, file_field: str) -> Path:
    """
    Resolve a stored file path (relative to audio_dir or bare filename) to
    an absolute Path using the cached directory index.
    """
    audio_dir = Path(audio_dir)
    # Try direct relative path first
    candidate = audio_dir / file_field
    if candidate.exists():
        return candidate
    # Fall back to index lookup by filename
    _ensure_audio_index(audio_dir)
    name = Path(file_field).name
    return _audio_index.get(name, candidate)


# ── Feature loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_features(parquet_path) -> pd.DataFrame:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        st.error(f"features.parquet not found at {parquet_path}")
        st.stop()
    return pd.read_parquet(parquet_path)


# ── Cosine similarity ─────────────────────────────────────────────────────────
def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarities between query (1-D) and each row of corpus (2-D)."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10
    return (corpus / norms) @ query_norm


# ── Playlist export ───────────────────────────────────────────────────────────
def export_m3u8(tracks: pd.DataFrame, audio_dir: Path = DEFAULT_AUDIO_DIR) -> str:
    lines = ["#EXTM3U"]
    for _, row in tracks.iterrows():
        filepath = find_audio_file(audio_dir, row["file"])
        lines.append(f"#EXTINF:-1,{row.get('track_id', row['file'])}")
        lines.append(str(filepath.resolve()))
    return "\n".join(lines)


# ── Genre label helper ────────────────────────────────────────────────────────
def top_genre_label(genre_vec, labels) -> str:
    if genre_vec is None or not labels:
        return ""
    try:
        arr = np.array(genre_vec)
        return labels[int(np.argmax(arr))]
    except Exception:
        return ""


# ── Rendering helpers ─────────────────────────────────────────────────────────
def _build_label(row: pd.Series, prefix: str = "", genre_label: str = "") -> str:
    parts = [f"{prefix}**{row.get('track_id', row['file'])}**"]
    if genre_label:
        parts.append(f"<br>🎵 *{genre_label}*")
    if pd.notna(row.get("bpm")):
        parts.append(f"BPM: {row['bpm']:.0f}")
    if pd.notna(row.get("danceability")):
        parts.append(f"Dance: {row['danceability']:.2f}")
    if pd.notna(row.get("voice_prob")):
        parts.append("Vocal" if row["voice_prob"] >= 0.5 else "Instr.")
    if pd.notna(row.get("key_temperley")) and pd.notna(row.get("scale_temperley")):
        parts.append(f"{row['key_temperley']} {row['scale_temperley']}")
    if pd.notna(row.get("loudness_lufs")):
        parts.append(f"{row['loudness_lufs']:.1f} LUFS")
    return " | ".join(parts)


def render_audio_player(
    track_row: pd.Series,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    show_info: bool = True,
    genre_label: str = "",
):
    filepath = find_audio_file(audio_dir, track_row["file"])
    if show_info:
        st.markdown(_build_label(track_row, genre_label=genre_label))
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
    labels=None,
):
    for _, row in df.head(n).iterrows():
        prefix = (
            f"Sim: {row[sim_col]:.4f} | "
            if (show_similarity and sim_col and sim_col in row.index)
            else ""
        )
        genre_label = top_genre_label(row.get("genre_discogs400"), labels or [])
        st.markdown(_build_label(row, prefix=prefix, genre_label=genre_label))
        filepath = find_audio_file(audio_dir, row["file"])
        if filepath.exists():
            st.audio(str(filepath))
        else:
            st.caption(f"File not found: {filepath}")
        st.divider()
