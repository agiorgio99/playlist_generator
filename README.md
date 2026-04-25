# Music Playlist Generator

Audio analysis and playlist generation toolkit built on **Essentia**, **Discogs-Effnet**, and **LAION-CLAP**. Given any folder of audio files, the pipeline extracts tempo, key, loudness, genre, danceability, voice/instrumental classification, and two embedding spaces (timbral and semantic), then exposes three interactive Streamlit applications for descriptor-based filtering, audio similarity search, and free-text music retrieval.

---

## Portability

The project is fully self-contained and runs on any machine without cloud dependencies or hardcoded paths. On first run, `analyze_collection.py` interactively asks for your Hugging Face token and the path to your audio folder, validates both, and downloads all required model weights automatically into the local `models/` folder. After that, every subsequent run and every Streamlit app works fully offline. Results are stored in `analysis_results/<collection_name>/` as a Parquet file, and all three apps scan that folder at startup to discover every analysed collection automatically — no configuration file needs to be edited.

This means you can copy the entire `playlist_generator/` folder to a different computer, run `analyze_collection.py` once to re-download models and re-analyse your audio, and immediately use all three apps on the new machine.

---

## Repository structure

```
playlist_generator/
├── analyze_collection.py     # Interactive analysis script — extracts all features
├── report.py                 # Interactive report generator — plots + HTML report
├── app_descriptors.py        # Streamlit app 1 — descriptor-based playlist filtering
├── app_similarity.py         # Streamlit app 2 — audio similarity search
├── app_text_query.py         # Streamlit app 3 — free-text CLAP search
├── utils.py                  # Shared utilities (data loading, similarity, audio index)
├── environment.yml           # Conda environment definition (Python 3.10)
├── requirements.txt          # Pip dependencies
├── config.yaml               # Optional overrides (models dir, results base path)
├── models/                   # Downloaded model weights (auto-populated on first run)
│   ├── discogs-effnet-bs64-1.pb
│   ├── genre_discogs400-discogs-effnet-1.pb
│   ├── genre_discogs400-discogs-effnet-1.json
│   ├── voice_instrumental-discogs-effnet-1.pb
│   ├── danceability-discogs-effnet-1.pb
│   └── music_speech_epoch_15_esc_89.25.pt   (LAION-CLAP checkpoint)
├── analysis_results/         # One subfolder per analysed collection
│   ├── Collection1/
│   │   ├── features.parquet
│   │   └── collection_info.json
│   └── Collection2/
│       ├── features.parquet
│       └── collection_info.json
└── reports/                  # Generated reports, one subfolder per collection
    ├── report_Collection1/
    │   ├── report.html
    │   └── figures/
    └── report_Collection2/
        ├── report.html
        └── figures/
```

---

## Requirements

- **Python 3.10** (required by Essentia TensorFlow)
- **Conda** (recommended) or a plain Python 3.10 virtual environment
- **Hugging Face account** — a free read token is needed to download the CLAP checkpoint on first run. Create one at https://huggingface.co/settings/tokens
- A folder of audio files in any of the following formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.m4a`, `.wma`, `.opus`

### Collections used in this project

- **MusAV** — available at https://drive.google.com/drive/folders/197MdMGGVGxqo3dSesk4Iln4N1pVyt1SX. Download and place the `audio_chunks/` folder in the repo root.
- **Royalty-Free Audio Dataset** — available on Kaggle at https://www.kaggle.com/datasets/. Download and place the audio files in a folder of your choice (the analysis script will ask you where).

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate musav-env
```

If you prefer pip without conda, use Python 3.10 and run:

```bash
pip install -r requirements.txt
```

> **Note:** `essentia-tensorflow` only has wheels for Python 3.10 on Linux and macOS. On Windows, run the project inside WSL2 or use the conda environment, which handles the platform differences.

### 2. Get a Hugging Face token

Go to https://huggingface.co/settings/tokens, create a token with **read** permissions, and copy it. The analysis script will ask for it on first run and validate it automatically. You do not need to save it anywhere — it is only used to download the CLAP model and is not stored after the session ends.

---

## Step-by-step usage

### Step 1 — Analyse a collection

```bash
python analyze_collection.py
```

The script runs interactively and will:

1. Ask for your **Hugging Face token** and validate it against the API.
2. Download all five model weights into `models/` if they are not already present (this takes a few minutes on first run).
3. Ask for the **path to your audio folder**, e.g. `./audio_chunks` or `/home/user/music/jazz_collection`. It validates that the folder exists and contains at least one supported audio file before starting.
4. Run the full extraction pipeline on every audio file found (recursively), showing a progress bar.
5. Save results to `analysis_results/<folder_name>/features.parquet` and `analysis_results/<folder_name>/collection_info.json`.

To skip the model download check if models are already present:

```bash
python analyze_collection.py --skip-download
```

Expected runtime: roughly 2 to 5 seconds per track depending on hardware. A collection of 2,000 tracks takes approximately 2 to 3 hours on a standard laptop CPU.

### Step 2 — Generate a report

```bash
python report.py
```

The script will:

1. List all analysed collections found in `analysis_results/` with their track count and analysis date.
2. Ask you to select one by number.
3. Ask for a report name (default: `report_<collection_name>`).
4. Generate distribution plots for all extracted features and save them to `reports/<report_name>/figures/`.
5. Produce a self-contained `reports/<report_name>/report.html` that you can open in any browser.

To skip the interactive prompts and run non-interactively:

```bash
python report.py --collection audio_chunks --report-name my_report
```

### Step 3 — Launch the Streamlit apps

All three apps can be run independently. Open a separate terminal for each one, or run them on different ports.

```bash
# App 1 — filter playlists by tempo, key, loudness, genre, danceability, voice
streamlit run app_descriptors.py

# App 2 — find tracks that sound similar to a reference track
streamlit run app_similarity.py

# App 3 — search tracks with a free-text description
streamlit run app_text_query.py
```

Each app opens in your browser at `http://localhost:8501` by default. To run multiple apps at the same time, specify a port for each:

```bash
streamlit run app_descriptors.py --server.port 8501
streamlit run app_similarity.py  --server.port 8502
streamlit run app_text_query.py  --server.port 8503
```

On first load, each app scans `analysis_results/` for valid collections and shows a **sidebar dropdown** to select which one to browse. No configuration is needed — any collection analysed with `analyze_collection.py` will appear automatically.

---

## Adding your own collection

1. Place your audio files in any folder on your computer (no need to move them into the project).
2. Run `python analyze_collection.py` and enter the path when prompted.
3. Once analysis is complete, restart (or reload) any Streamlit app — your collection will appear in the sidebar dropdown alongside the existing ones.

Supported audio formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.m4a`, `.wma`, `.opus`

---

## How the apps work

### App 1 — Descriptor-Based Playlist (`app_descriptors.py`)

Filters the selected collection using sliders and dropdowns for:

- **Tempo** (BPM range)
- **Key** (C, C#, D, ... B)
- **Scale** (major / minor)
- **Loudness** (LUFS range)
- **Genre** (Discogs top-level family)
- **Danceability** (0 to 1 score)
- **Voice / Instrumental** toggle

Results are displayed as a table with track names, feature values, and an in-browser audio player. The filtered list can be exported as an M3U8 playlist file.

### App 2 — Audio Similarity (`app_similarity.py`)

Select any track from the collection as a seed and retrieve the *k* most similar tracks. Similarity is computed via cosine distance on the **1,280-dimensional Discogs-Effnet embedding** (captures timbral and production-style similarity). An optional toggle switches to **CLAP-based re-ranking**, which adds semantic similarity on top of timbral matching.

### App 3 — Text Query (`app_text_query.py`)

Type a free-text description (e.g. *"melancholic ambient music"*, *"energetic dance track with a fast beat"*) and retrieve the top matching tracks. Preset example buttons trigger a search immediately without typing. Retrieval is done by encoding the text with the LAION-CLAP text encoder and finding the nearest audio CLAP embeddings by cosine similarity.

---

## Feature details

| Feature | Extractor | Notes |
|---|---|---|
| Tempo (BPM) | `RhythmExtractor2013` | Accurate on rhythmically clear genres; occasional half/double-time errors on complex material |
| Key & scale | `KeyExtractor` | Reliable on tonal music; degrades on distorted or atonal tracks |
| Loudness (LUFS) | `LoudnessEBUR128` | Highly reliable across all genres |
| Genre (400 classes) | Discogs-Effnet classifier | Correct at family level; occasional sub-genre confusion |
| Danceability | Effnet-based head | Can overestimate for fast non-dance genres (e.g. Punk) |
| Voice / Instrumental | Effnet-based head | Reliable on standard pop and rock; less so on heavily produced electronic music |
| Timbral embedding | Discogs-Effnet (1,280-d) | Used for similarity search |
| Semantic embedding | LAION-CLAP (512-d) | Used for text queries and optional re-ranking |

---

## Troubleshooting

**`essentia-tensorflow` import error on Windows**
Run inside WSL2 with a Linux Python 3.10 environment, or use the conda environment which patches the binary dependencies automatically.

**`[ WARNING ] No network created...` in the terminal during analysis**
This is a harmless internal Essentia message that has been suppressed. If you still see it, verify that `essentia.log.warningActive = False` is set at the top of `analyze_collection.py`.

**The app shows "No collections found"**
This means `analysis_results/` is empty or contains no valid `features.parquet` files. Run `analyze_collection.py` on at least one audio folder first.

**Analysis is very slow**
The pipeline runs on CPU by default. GPU acceleration is not supported by `essentia-tensorflow` in this configuration. Running overnight on a large collection (2,000+ tracks) is normal.
