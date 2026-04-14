# MusAV Analysis & Playlist Generation

Audio analysis and playlist generation toolkit built on Essentia, Discogs-Effnet, and CLAP. Extracts tempo, key, loudness, genre, danceability, voice/instrumental classification, and embedding vectors from a music collection, then provides interactive Streamlit apps for descriptor-based playlist generation, track similarity search, and free-text music retrieval.

> **Status:** Work in progress.

## Repo structure

| File | Description |
|---|---|
| `environment.yml` | Conda environment definition (Python 3.10) |
| `requirements.txt` | Pip dependencies |
| `analyze_collection.py` | Batch audio analysis — extracts all features and saves JSON + parquet |
| `report.py` | Generates distribution plots and summary statistics from analysis results |
| `app_descriptors.py` | Streamlit app — filter & build playlists by style, tempo, key, etc. |
| `app_similarity.py` | Streamlit app — find similar tracks via Effnet and CLAP embeddings |
| `app_text_query.py` | Streamlit app — search tracks with free-text descriptions (CLAP) |
| `utils.py` | Shared utilities (data loading, similarity, M3U8 export, audio players) |

## Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate musav-env

# 2. Download models into ./models/
#    - discogs-effnet-bs64-1.pb
#    - genre_discogs400-discogs-effnet-1.pb
#    - voice_instrumental-discogs-effnet-1.pb
#    - danceability-discogs-effnet-1.pb
#    - music_speech_epoch_15_esc_89.25.pt  (CLAP checkpoint)
#    See https://essentia.upf.edu/models.html for Essentia models.

# 3. Place audio files in ./audio_chunks/

# 4. Run analysis
python analyze_collection.py

# 5. Generate report
python report.py

# 6. Launch Streamlit apps
streamlit run app_descriptors.py
streamlit run app_similarity.py
streamlit run app_text_query.py
```
