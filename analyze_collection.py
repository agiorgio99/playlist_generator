#\!/usr/bin/env python3
"""
analyze_collection.py - Analyze a music collection using Essentia and CLAP.

Interactive at startup: asks for HuggingFace token and audio directory.
Downloads missing models automatically. Results are saved in a per-collection
subfolder so multiple collections never overwrite each other.

Run:  python analyze_collection.py
      python analyze_collection.py --audio-dir /path/to/music --limit 10
"""

import argparse
import getpass
import json
import logging
import os
import sys
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Silence Essentia's "No network created" scheduler warnings ───────────────
import essentia
essentia.log.warningActive = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG — reads config.yaml first, CLI flags override everything ──────────
def _load_yaml_config(config_path: Path = Path("config.yaml")) -> dict:
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Warning: could not read {config_path}: {exc}")
        return {}


_yaml = _load_yaml_config()

CONFIG = {
    "audio_dir":          Path(_yaml.get("audio_dir",        "./audio_chunks")),
    "results_base":       Path(_yaml.get("results_base",     "./analysis_results")),
    "models_dir":         Path(_yaml.get("models_dir",       "./models")),
    "effnet_model":       "discogs-effnet-bs64-1.pb",
    "genre_model":        "genre_discogs400-discogs-effnet-1.pb",
    "voice_model":        "voice_instrumental-discogs-effnet-1.pb",
    "danceability_model": "danceability-discogs-effnet-1.pb",
    "clap_checkpoint":    _yaml.get("clap_checkpoint", "music_speech_epoch_15_esc_89.25.pt"),
    "sr_essentia_tf":     16000,
    "sr_clap":            48000,
}

# Audio extensions to search for
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}

# Essentia model download URLs (no auth required)
ESSENTIA_MODEL_URLS = {
    "discogs-effnet-bs64-1.pb":
        "https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb",
    "genre_discogs400-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/music-style-classification/discogs-effnet/genre_discogs400-discogs-effnet-1.pb",
    "genre_discogs400-discogs-effnet-1.json":
        "https://essentia.upf.edu/models/music-style-classification/discogs-effnet/genre_discogs400-discogs-effnet-1.json",
    "voice_instrumental-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb",
    "danceability-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb",
}

CLAP_HF_REPO     = "lukewys/laion_clap"
CLAP_FILENAME    = "music_speech_epoch_15_esc_89.25.pt"


# ── HuggingFace token setup ──────────────────────────────────────────────────
def setup_hf_token() -> str:
    """
    Interactively ask for a HuggingFace token, validate it, set env vars.
    Returns the valid token string.
    """
    print()
    print("=" * 62)
    print("  HUGGING FACE TOKEN REQUIRED")
    print("=" * 62)
    print()
    print("  The CLAP audio-language model is hosted on HuggingFace and")
    print("  requires a free Read token to download.")
    print()
    print("  How to create a token:")
    print("  1. Open  https://huggingface.co/settings/tokens")
    print("  2. Click 'New token'  →  Role: Read  →  give it any name")
    print("  3. Copy the token (it starts with  hf_...)")
    print()

    while True:
        token = getpass.getpass("  Paste your token here (hidden): ").strip()
        if not token:
            print("  ✗ Token cannot be empty. Try again.\n")
            continue
        if not token.startswith("hf_"):
            print("  ✗ Token should start with 'hf_'. Try again.\n")
            continue

        print("  Validating token …", end=" ", flush=True)
        try:
            from huggingface_hub import whoami
            info = whoami(token=token)
            print(f"✓  (logged in as: {info['name']})")
            print()
        except Exception as exc:
            print(f"\n  ✗ Token rejected: {exc}")
            print("  Please check the token and try again.\n")
            continue

        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return token


# ── Model download helpers ───────────────────────────────────────────────────
class _TqdmHook(tqdm):
    """Progress-bar hook for urllib.request.urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _TqdmHook(unit="B", unit_scale=True, miniters=1, desc=dest.name, ncols=72) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def ensure_models(models_dir: Path, hf_token: str) -> None:
    """Download any missing Essentia models and CLAP checkpoint."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Checking Essentia models …")
    for filename, url in ESSENTIA_MODEL_URLS.items():
        dest = models_dir / filename
        if dest.exists():
            print(f"  ✓  {filename}")
        else:
            print(f"  ↓  {filename}")
            try:
                _download_url(url, dest)
            except Exception as exc:
                log.error("Failed to download %s: %s", filename, exc)

    print("Checking CLAP checkpoint …")
    clap_dest = models_dir / CLAP_FILENAME
    if clap_dest.exists():
        print(f"  ✓  {CLAP_FILENAME}")
    else:
        print(f"  ↓  {CLAP_FILENAME}  (from HuggingFace: {CLAP_HF_REPO})")
        try:
            from huggingface_hub import hf_hub_download
            tmp = hf_hub_download(
                repo_id=CLAP_HF_REPO,
                filename=CLAP_FILENAME,
                token=hf_token,
                local_dir=str(models_dir),
            )
            # hf_hub_download may place the file in a cache subdir; move it
            tmp = Path(tmp)
            if tmp != clap_dest:
                import shutil
                shutil.copy2(tmp, clap_dest)
            print(f"  ✓  {CLAP_FILENAME}")
        except Exception as exc:
            log.error(
                "Could not download CLAP checkpoint: %s\n"
                "Download manually from https://huggingface.co/%s and place it in %s",
                exc, CLAP_HF_REPO, models_dir,
            )
    print()


# ── Audio directory setup ────────────────────────────────────────────────────
def setup_audio_dir(default: Path) -> Path:
    """Ask the user for an audio directory; loops until a valid one is given."""
    print(f"  Default audio directory:  {default}")
    while True:
        raw = input(f"  Audio directory [{default}]: ").strip()
        audio_dir = Path(raw) if raw else default

        if not audio_dir.exists():
            print(f"\n  ✗  Directory not found: {audio_dir}")
            print("     Please enter a valid path and try again.\n")
            continue

        files = find_audio_files(audio_dir)
        if not files:
            print(
                f"\n  ✗  No audio files found in {audio_dir}\n"
                f"     (searched for: {', '.join(sorted(AUDIO_EXTENSIONS))})\n"
                "     Make sure the folder contains audio files and try again.\n"
            )
            continue

        print(f"  ✓  Found {len(files)} audio file(s) in {audio_dir}\n")
        return audio_dir


def find_audio_files(audio_dir: Path) -> list:
    """Return all audio files under audio_dir (any supported format)."""
    found = []
    for ext in AUDIO_EXTENSIONS:
        found.extend(audio_dir.rglob(f"*{ext}"))
        found.extend(audio_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(found))


# ── Audio loading & resampling ───────────────────────────────────────────────
def load_audio(filepath: Path):
    from essentia.standard import AudioLoader, MonoMixer
    stereo, sr, n_channels, _, _, _ = AudioLoader(filename=str(filepath))()
    mono = MonoMixer()(stereo, n_channels)
    return mono, stereo, sr


def resample(mono: np.ndarray, sr_in: float, sr_out: int) -> np.ndarray:
    from essentia.standard import Resample
    return Resample(inputSampleRate=float(sr_in), outputSampleRate=sr_out)(mono)


# ── Feature extractors ───────────────────────────────────────────────────────
def extract_tempo(mono: np.ndarray) -> dict:
    from essentia.standard import RhythmExtractor2013
    bpm, _, confidence, _, _ = RhythmExtractor2013()(mono)
    return {"bpm": float(bpm), "confidence": float(confidence)}


def extract_key(mono: np.ndarray) -> dict:
    from essentia.standard import KeyExtractor
    results = {}
    for profile in ("temperley", "krumhansl", "edma"):
        key, scale, strength = KeyExtractor(profileType=profile)(mono)
        results[profile] = {"key": key, "scale": scale, "strength": float(strength)}
    return results


def extract_loudness(stereo: np.ndarray) -> dict:
    from essentia.standard import LoudnessEBUR128
    _, _, integrated, _ = LoudnessEBUR128()(stereo)
    return {"integrated_lufs": float(integrated)}


def extract_clap_embedding(mono_48k: np.ndarray, clap_model) -> np.ndarray:
    import torch
    audio_tensor = torch.from_numpy(mono_48k).unsqueeze(0)
    with torch.no_grad():
        emb = clap_model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
    return emb.squeeze(0).cpu().numpy()


# ── TensorflowPredict2D node-name auto-detection ─────────────────────────────
_PREDICT2D_CANDIDATES = [
    ("serving_default_model_Placeholder", "PartitionedCall"),
    ("model/Placeholder", "model/Softmax"),
    ("model/Placeholder", "model/Sigmoid"),
]


def make_predict2d(model_path: str):
    from essentia.standard import TensorflowPredict2D
    last_exc = None
    for inp, out in _PREDICT2D_CANDIDATES:
        try:
            return TensorflowPredict2D(graphFilename=model_path, input=inp, output=out)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"Could not configure TensorflowPredict2D for {model_path}. Last error: {last_exc}"
    )


# ── Model cache ──────────────────────────────────────────────────────────────
class ModelCache:
    def __init__(self, cfg: dict):
        models_dir = cfg["models_dir"]
        effnet_path = str(models_dir / cfg["effnet_model"])
        genre_path  = str(models_dir / cfg["genre_model"])
        voice_path  = str(models_dir / cfg["voice_model"])
        dance_path  = str(models_dir / cfg["danceability_model"])

        self.effnet_runner = None
        if Path(effnet_path).exists():
            from essentia.standard import TensorflowPredictEffnetDiscogs
            self.effnet_runner = TensorflowPredictEffnetDiscogs(
                graphFilename=effnet_path, output="PartitionedCall:1"
            )
        else:
            log.warning("Effnet model not found at %s.", effnet_path)

        self.genre_runner = make_predict2d(genre_path) if Path(genre_path).exists() else None
        self.voice_runner = make_predict2d(voice_path) if Path(voice_path).exists() else None
        self.dance_runner = make_predict2d(dance_path) if Path(dance_path).exists() else None

        self.clap_model = None
        clap_ckpt = models_dir / cfg["clap_checkpoint"]
        if clap_ckpt.exists():
            try:
                import laion_clap
                self.clap_model = laion_clap.CLAP_Module(
                    enable_fusion=False, amodel="HTSAT-base", device="cpu"
                )
                self.clap_model.load_ckpt(str(clap_ckpt))
                log.info("CLAP model loaded from %s.", clap_ckpt)
            except Exception as exc:
                log.warning("Failed to load CLAP model: %s", exc)
        else:
            log.warning("CLAP checkpoint not found at %s — CLAP embeddings disabled.", clap_ckpt)


# ── Per-track analysis ───────────────────────────────────────────────────────
def analyze_track(filepath: Path, cache: ModelCache, cfg: dict) -> dict:
    mono, stereo, sr = load_audio(filepath)

    result = {}
    result["tempo"]    = extract_tempo(mono)
    result["key"]      = extract_key(mono)
    result["loudness"] = extract_loudness(stereo)

    mono_16k = resample(mono, sr, cfg["sr_essentia_tf"])

    if cache.effnet_runner is not None:
        raw_embs = cache.effnet_runner(mono_16k)
        result["effnet_embedding"] = np.mean(raw_embs, axis=0).tolist()

        if cache.genre_runner is not None:
            genre_preds = cache.genre_runner(raw_embs)
            result["genre_discogs400"] = np.mean(genre_preds, axis=0).tolist()

        if cache.voice_runner is not None:
            voice_mean = np.mean(cache.voice_runner(raw_embs), axis=0)
            result["voice_prob"] = float(voice_mean[1]) if len(voice_mean) > 1 else float(voice_mean[0])

        if cache.dance_runner is not None:
            dance_mean = np.mean(cache.dance_runner(raw_embs), axis=0)
            result["danceability"] = float(dance_mean[1]) if len(dance_mean) > 1 else float(dance_mean[0])

    if cache.clap_model is not None:
        mono_48k = resample(mono, sr, cfg["sr_clap"])
        result["clap_embedding"] = extract_clap_embedding(mono_48k, cache.clap_model).tolist()

    return result


# ── JSON → Parquet merge ─────────────────────────────────────────────────────
def merge_to_parquet(results_dir: Path) -> pd.DataFrame:
    records = []
    for jf in sorted(results_dir.glob("*.json")):
        if jf.name == "collection_info.json":
            continue
        try:
            text = jf.read_text().strip()
            if not text:
                continue
            data = json.loads(text)
            if not data:
                continue
        except Exception:
            continue

        flat = {
            "track_id":       jf.stem,
            "file":           data.get("file", ""),
            "bpm":            data.get("tempo", {}).get("bpm"),
            "bpm_confidence": data.get("tempo", {}).get("confidence"),
            "loudness_lufs":  data.get("loudness", {}).get("integrated_lufs"),
            "voice_prob":     data.get("voice_prob"),
            "danceability":   data.get("danceability"),
        }
        for profile in ("temperley", "krumhansl", "edma"):
            kd = data.get("key", {}).get(profile, {})
            flat[f"key_{profile}"]          = kd.get("key")
            flat[f"scale_{profile}"]        = kd.get("scale")
            flat[f"key_strength_{profile}"] = kd.get("strength")

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


# ── CLI & main ───────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a music collection.")
    parser.add_argument("--audio-dir",  type=Path, default=None,
                        help="Root folder containing audio files (skips interactive prompt).")
    parser.add_argument("--models-dir", type=Path, default=CONFIG["models_dir"],
                        help="Folder containing Essentia .pb models and CLAP checkpoint.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N tracks (useful for testing).")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip model download check (use if models already present).")
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = args.models_dir

    # ── Step 1: HuggingFace token ────────────────────────────────────────────
    if not args.skip_download:
        hf_token = setup_hf_token()
    else:
        hf_token = os.environ.get("HF_TOKEN", "")

    # ── Step 2: Download missing models ─────────────────────────────────────
    if not args.skip_download:
        ensure_models(models_dir, hf_token)

    # ── Step 3: Audio directory ──────────────────────────────────────────────
    print("=" * 62)
    print("  AUDIO COLLECTION")
    print("=" * 62)
    if args.audio_dir is not None:
        audio_dir = args.audio_dir
        audio_files = find_audio_files(audio_dir)
        if not audio_files:
            print(f"\n  ✗  No audio files found in {audio_dir}")
            sys.exit(1)
        print(f"  ✓  {len(audio_files)} audio file(s) in {audio_dir}\n")
    else:
        audio_dir = setup_audio_dir(CONFIG["audio_dir"])

    # ── Results subfolder named after the collection ─────────────────────────
    collection_name = audio_dir.resolve().name
    results_dir = CONFIG["results_base"] / collection_name
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        **CONFIG,
        "audio_dir":   audio_dir,
        "results_dir": results_dir,
        "models_dir":  models_dir,
        "limit":       args.limit,
    }

    # Save collection metadata so apps can locate audio files later
    info = {
        "collection_name": collection_name,
        "audio_dir":       str(audio_dir.resolve()),
        "analyzed_at":     datetime.now().isoformat(timespec="seconds"),
    }
    (results_dir / "collection_info.json").write_text(json.dumps(info, indent=2))

    # ── Step 4: Find tracks to process ──────────────────────────────────────
    all_audio = find_audio_files(audio_dir)
    log.info("Found %d audio file(s) in %s", len(all_audio), audio_dir)

    done = set()
    for jf in results_dir.glob("*.json"):
        if jf.name == "collection_info.json":
            continue
        try:
            text = jf.read_text().strip()
            if text and json.loads(text):
                done.add(jf.stem)
        except Exception:
            pass

    todo = [f for f in all_audio if f.stem not in done]
    if cfg.get("limit"):
        todo = todo[: cfg["limit"]]
        log.info("--limit %d: processing first %d tracks.", cfg["limit"], len(todo))
    log.info("%d already analyzed, %d remaining.", len(done), len(todo))

    # ── Step 5: Load models and run analysis ─────────────────────────────────
    cache = ModelCache(cfg)

    for filepath in tqdm(todo, desc="Analyzing", unit="track"):
        out_json = results_dir / f"{filepath.stem}.json"
        try:
            result = analyze_track(filepath, cache, cfg)
            result["file"] = str(filepath.relative_to(audio_dir))
            with open(out_json, "w") as f:
                json.dump(result, f)
        except Exception as exc:
            log.warning("Skipping %s: %s", filepath.name, exc)
            traceback.print_exc()

    merge_to_parquet(results_dir)
    log.info("Done. Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
