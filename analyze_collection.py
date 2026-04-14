#!/usr/bin/env python3
"""
analyze_collection.py — Analyze a music collection using Essentia and CLAP.

Extracts tempo, key, loudness, genre, danceability, voice/instrumental,
Discogs-Effnet embeddings, and CLAP embeddings for each track.
Saves per-track JSON results and a combined features.parquet.
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# CONFIG — all paths centralised here; override via CLI flags
# ──────────────────────────────────────────────────────────────────────
CONFIG = {
    "audio_dir": Path("./audio_chunks"),
    "results_dir": Path("./analysis_results"),
    "models_dir": Path("./models"),
    # Essentia TF model filenames (expected inside models_dir)
    "effnet_model": "discogs-effnet-bs64-1.pb",
    "genre_model": "genre_discogs400-discogs-effnet-1.pb",
    "voice_model": "voice_instrumental-discogs-effnet-1.pb",
    "danceability_model": "danceability-discogs-effnet-1.pb",
    # CLAP checkpoint
    "clap_checkpoint": "music_speech_epoch_15_esc_89.25.pt",
    # Sample rates
    "sr_essentia_tf": 16000,
    "sr_clap": 48000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Helper: load & resample audio
# ──────────────────────────────────────────────────────────────────────
def load_audio(filepath: Path):
    """Load an audio file and return mono/stereo at original SR.

    Returns
    -------
    mono : np.ndarray   (float32, range [-1, 1])
    stereo : np.ndarray  (float32, shape [N, 2])
    sr : float
    """
    from essentia.standard import AudioLoader, MonoMixer

    loader = AudioLoader(filename=str(filepath))
    stereo, sr, n_channels, _, _, _ = loader()

    mixer = MonoMixer(numberChannels=n_channels)
    mono = mixer(stereo)

    return mono, stereo, sr


def resample(mono: np.ndarray, sr_in: float, sr_out: int) -> np.ndarray:
    """Resample a mono signal to *sr_out* Hz."""
    from essentia.standard import Resample

    resampler = Resample(inputSampleRate=sr_in, outputSampleRate=sr_out)
    return resampler(mono)


# ──────────────────────────────────────────────────────────────────────
# Feature extractors
# ──────────────────────────────────────────────────────────────────────
def extract_tempo(mono: np.ndarray) -> dict:
    """Extract BPM and confidence using RhythmExtractor2013."""
    from essentia.standard import RhythmExtractor2013

    rhythm = RhythmExtractor2013()
    bpm, ticks, confidence, estimates, intervals = rhythm(mono)
    return {"bpm": float(bpm), "confidence": float(confidence)}


def extract_key(mono: np.ndarray) -> dict:
    """Extract key/scale with three profiles (temperley, krumhansl, edma)."""
    from essentia.standard import KeyExtractor

    results = {}
    for profile in ("temperley", "krumhansl", "edma"):
        extractor = KeyExtractor(profileType=profile)
        key, scale, strength = extractor(mono)
        results[profile] = {
            "key": key,
            "scale": scale,
            "strength": float(strength),
        }
    return results


def extract_loudness(stereo: np.ndarray) -> dict:
    """Compute integrated loudness (LUFS) via EBU R128."""
    from essentia.standard import LoudnessEBUR128

    loudness = LoudnessEBUR128()
    integrated, _, _, _ = loudness(stereo)
    return {"integrated_lufs": float(integrated)}


def extract_effnet_embeddings(mono_16k: np.ndarray, model_path: str) -> np.ndarray:
    """Run Discogs-Effnet and return mean embedding (1280-d)."""
    from essentia.standard import TensorflowPredictEffnetDiscogs

    model = TensorflowPredictEffnetDiscogs(
        graphFilename=model_path, output="PartitionedCall:1"
    )
    embeddings = model(mono_16k)
    return np.mean(embeddings, axis=0)


def extract_genre(embeddings_2d: np.ndarray, model_path: str) -> np.ndarray:
    """Predict Discogs-400 genre activations from Effnet embeddings."""
    from essentia.standard import TensorflowPredict2D

    model = TensorflowPredict2D(graphFilename=model_path)
    preds = model(embeddings_2d)
    return np.mean(preds, axis=0)


def extract_voice_instrumental(mono_16k: np.ndarray, model_path: str, effnet_path: str) -> float:
    """Return voice probability (0 = instrumental, 1 = vocal)."""
    from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D

    effnet = TensorflowPredictEffnetDiscogs(
        graphFilename=effnet_path, output="PartitionedCall:1"
    )
    embeddings = effnet(mono_16k)

    classifier = TensorflowPredict2D(graphFilename=model_path)
    preds = classifier(embeddings)
    mean_preds = np.mean(preds, axis=0)
    # Index 0 = instrumental, Index 1 = voice (convention)
    return float(mean_preds[0]) if len(mean_preds) == 1 else float(mean_preds[1])


def extract_danceability(mono_16k: np.ndarray, model_path: str, effnet_path: str) -> float:
    """Return danceability probability."""
    from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D

    effnet = TensorflowPredictEffnetDiscogs(
        graphFilename=effnet_path, output="PartitionedCall:1"
    )
    embeddings = effnet(mono_16k)

    classifier = TensorflowPredict2D(graphFilename=model_path)
    preds = classifier(embeddings)
    mean_preds = np.mean(preds, axis=0)
    return float(mean_preds[1]) if len(mean_preds) > 1 else float(mean_preds[0])


def extract_clap_embedding(mono_48k: np.ndarray, clap_model) -> np.ndarray:
    """Return mean CLAP audio embedding."""
    import torch

    audio_tensor = torch.from_numpy(mono_48k).unsqueeze(0)
    with torch.no_grad():
        emb = clap_model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
    return emb.squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────
# Model loader (instantiate once)
# ──────────────────────────────────────────────────────────────────────
class ModelCache:
    """Holds heavy model objects so they are instantiated only once."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.models_dir = cfg["models_dir"]

        # Essentia TF models — lazy-checked paths
        self.effnet_path = str(self.models_dir / cfg["effnet_model"])
        self.genre_path = str(self.models_dir / cfg["genre_model"])
        self.voice_path = str(self.models_dir / cfg["voice_model"])
        self.dance_path = str(self.models_dir / cfg["danceability_model"])

        self.effnet_ok = Path(self.effnet_path).exists()
        self.genre_ok = Path(self.genre_path).exists()
        self.voice_ok = Path(self.voice_path).exists()
        self.dance_ok = Path(self.dance_path).exists()

        if not self.effnet_ok:
            log.warning("Effnet model not found at %s — Effnet features disabled.", self.effnet_path)
        if not self.genre_ok:
            log.warning("Genre model not found at %s — genre features disabled.", self.genre_path)
        if not self.voice_ok:
            log.warning("Voice model not found at %s — voice features disabled.", self.voice_path)
        if not self.dance_ok:
            log.warning("Danceability model not found at %s — danceability features disabled.", self.dance_path)

        # CLAP
        self.clap_model = None
        clap_ckpt = self.models_dir / cfg["clap_checkpoint"]
        if clap_ckpt.exists():
            try:
                import laion_clap

                self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
                self.clap_model.load_ckpt(str(clap_ckpt))
                log.info("CLAP model loaded.")
            except Exception as exc:
                log.warning("Failed to load CLAP model: %s", exc)
        else:
            log.warning("CLAP checkpoint not found at %s — CLAP features disabled.", clap_ckpt)


# ──────────────────────────────────────────────────────────────────────
# Per-track analysis
# ──────────────────────────────────────────────────────────────────────
def analyze_track(filepath: Path, cache: ModelCache, cfg: dict) -> dict:
    """Run full analysis pipeline on a single track.

    Parameters
    ----------
    filepath : Path
        Path to the MP3 file.
    cache : ModelCache
        Pre-loaded models.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    dict with all extracted features.
    """
    mono, stereo, sr = load_audio(filepath)

    result: dict = {"file": str(filepath.name)}

    # ---- Tempo ----
    result["tempo"] = extract_tempo(mono)

    # ---- Key ----
    result["key"] = extract_key(mono)

    # ---- Loudness ----
    result["loudness"] = extract_loudness(stereo)

    # ---- Resample for TF models ----
    mono_16k = resample(mono, sr, cfg["sr_essentia_tf"])

    # ---- Discogs-Effnet embeddings ----
    effnet_emb = None
    if cache.effnet_ok:
        from essentia.standard import TensorflowPredictEffnetDiscogs

        effnet_runner = TensorflowPredictEffnetDiscogs(
            graphFilename=cache.effnet_path, output="PartitionedCall:1"
        )
        raw_embs = effnet_runner(mono_16k)
        effnet_emb = np.mean(raw_embs, axis=0)
        result["effnet_embedding"] = effnet_emb.tolist()

        # ---- Genre Discogs-400 ----
        if cache.genre_ok:
            from essentia.standard import TensorflowPredict2D

            genre_runner = TensorflowPredict2D(graphFilename=cache.genre_path)
            genre_preds = genre_runner(raw_embs)
            genre_mean = np.mean(genre_preds, axis=0)
            result["genre_discogs400"] = genre_mean.tolist()

        # ---- Voice / Instrumental ----
        if cache.voice_ok:
            from essentia.standard import TensorflowPredict2D as TP2D_v

            voice_runner = TP2D_v(graphFilename=cache.voice_path)
            voice_preds = voice_runner(raw_embs)
            voice_mean = np.mean(voice_preds, axis=0)
            result["voice_prob"] = float(voice_mean[1]) if len(voice_mean) > 1 else float(voice_mean[0])

        # ---- Danceability ----
        if cache.dance_ok:
            from essentia.standard import TensorflowPredict2D as TP2D_d

            dance_runner = TP2D_d(graphFilename=cache.dance_path)
            dance_preds = dance_runner(raw_embs)
            dance_mean = np.mean(dance_preds, axis=0)
            result["danceability"] = float(dance_mean[1]) if len(dance_mean) > 1 else float(dance_mean[0])

    # ---- CLAP embeddings ----
    if cache.clap_model is not None:
        mono_48k = resample(mono, sr, cfg["sr_clap"])
        clap_emb = extract_clap_embedding(mono_48k, cache.clap_model)
        result["clap_embedding"] = clap_emb.tolist()

    return result


# ──────────────────────────────────────────────────────────────────────
# Merge results → parquet
# ──────────────────────────────────────────────────────────────────────
def merge_to_parquet(results_dir: Path) -> pd.DataFrame:
    """Read all per-track JSON files and combine into a DataFrame + parquet."""
    records = []
    for jf in sorted(results_dir.glob("*.json")):
        with open(jf, "r") as f:
            data = json.load(f)

        flat: dict = {
            "track_id": jf.stem,
            "file": data.get("file", ""),
            "bpm": data.get("tempo", {}).get("bpm"),
            "bpm_confidence": data.get("tempo", {}).get("confidence"),
            "loudness_lufs": data.get("loudness", {}).get("integrated_lufs"),
            "voice_prob": data.get("voice_prob"),
            "danceability": data.get("danceability"),
        }

        # Key profiles
        for profile in ("temperley", "krumhansl", "edma"):
            kd = data.get("key", {}).get(profile, {})
            flat[f"key_{profile}"] = kd.get("key")
            flat[f"scale_{profile}"] = kd.get("scale")
            flat[f"key_strength_{profile}"] = kd.get("strength")

        # Embeddings stored as lists → keep as list objects in the DataFrame
        if "effnet_embedding" in data:
            flat["effnet_embedding"] = data["effnet_embedding"]
        if "genre_discogs400" in data:
            flat["genre_discogs400"] = data["genre_discogs400"]
        if "clap_embedding" in data:
            flat["clap_embedding"] = data["clap_embedding"]

        records.append(flat)

    df = pd.DataFrame(records)
    out_path = results_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved %d tracks to %s", len(df), out_path)
    return df


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    """Parse command-line arguments, falling back to CONFIG defaults."""
    parser = argparse.ArgumentParser(description="Analyze a music collection.")
    parser.add_argument("--audio-dir", type=Path, default=CONFIG["audio_dir"],
                        help="Directory containing MP3 files.")
    parser.add_argument("--results-dir", type=Path, default=CONFIG["results_dir"],
                        help="Directory to save analysis results.")
    parser.add_argument("--models-dir", type=Path, default=CONFIG["models_dir"],
                        help="Directory containing model files.")
    return parser.parse_args()


def main():
    """Entry point: iterate over MP3s, analyse each, merge to parquet."""
    args = parse_args()
    cfg = {**CONFIG, "audio_dir": args.audio_dir, "results_dir": args.results_dir, "models_dir": args.models_dir}

    audio_dir = cfg["audio_dir"]
    results_dir = cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(audio_dir.rglob("*.mp3"))
    if not mp3_files:
        log.error("No MP3 files found in %s", audio_dir)
        sys.exit(1)
    log.info("Found %d MP3 files in %s", len(mp3_files), audio_dir)

    # Determine already-analyzed tracks for resumption
    done = {p.stem for p in results_dir.glob("*.json")}
    todo = [f for f in mp3_files if f.stem not in done]
    log.info("%d already analyzed, %d remaining.", len(done), len(todo))

    # Load models once
    cache = ModelCache(cfg)

    errors: list[str] = []

    for filepath in tqdm(todo, desc="Analyzing", unit="track"):
        track_id = filepath.stem
        out_json = results_dir / f"{track_id}.json"
        try:
            result = analyze_track(filepath, cache, cfg)
            with open(out_json, "w") as f:
                json.dump(result, f)
        except Exception as exc:
            msg = f"{track_id}: {exc}"
            log.error(msg)
            errors.append(msg)
            continue

    if errors:
        log.warning("%d tracks failed. Errors:\n%s", len(errors), "\n".join(errors))

    # Merge all JSON → parquet
    merge_to_parquet(results_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
