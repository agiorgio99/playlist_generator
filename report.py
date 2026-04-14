#!/usr/bin/env python3
"""
report.py — Generate analysis report figures from features.parquet.

Produces PNG plots for genre, tempo, danceability, key/scale, loudness,
and voice/instrumental distributions. Also exports styles_distribution.tsv.
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
CONFIG = {
    "results_dir": Path("./analysis_results"),
    "figures_dir": Path("./report_figures"),
}

# Discogs-400 label list (broad parent genres extracted from "genre---style" format)
# Loaded dynamically from the data.


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument("--results-dir", type=Path, default=CONFIG["results_dir"])
    parser.add_argument("--figures-dir", type=Path, default=CONFIG["figures_dir"])
    return parser.parse_args()


def load_discogs400_labels(models_dir: Path = Path("./models")) -> list[str]:
    """Try to load the Discogs-400 label list from a metadata file.

    Falls back to a numbered list if the file is not found.
    """
    label_file = models_dir / "genre_discogs400-discogs-effnet-1.json"
    if label_file.exists():
        import json

        with open(label_file) as f:
            meta = json.load(f)
        return meta.get("classes", [f"class_{i}" for i in range(400)])
    # Try a plain text file
    txt_file = models_dir / "discogs400_labels.txt"
    if txt_file.exists():
        return [l.strip() for l in txt_file.read_text().splitlines() if l.strip()]
    return [f"class_{i}" for i in range(400)]


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────
def plot_genre_distribution(df: pd.DataFrame, labels: list[str], figures_dir: Path):
    """Bar chart of broad (parent) genre counts + TSV of full style distribution."""
    if "genre_discogs400" not in df.columns or df["genre_discogs400"].isna().all():
        print("  ⚠ genre_discogs400 column missing or empty — skipping genre plot.")
        return

    genre_vecs = np.stack(df["genre_discogs400"].dropna().values)

    # Determine top style per track
    top_indices = np.argmax(genre_vecs, axis=1)

    # If we have real labels, parse parent genres from "genre---style"
    if len(labels) == genre_vecs.shape[1]:
        style_names = [labels[i] for i in top_indices]
    else:
        style_names = [f"class_{i}" for i in top_indices]

    # Full style distribution → TSV
    style_counts = Counter(style_names)
    style_df = pd.DataFrame(
        sorted(style_counts.items(), key=lambda x: -x[1]),
        columns=["style", "count"],
    )
    style_df.to_csv(figures_dir / "styles_distribution.tsv", sep="\t", index=False)
    print(f"  Saved styles_distribution.tsv ({len(style_df)} unique styles)")

    # Parent (broad) genres
    parent_names = [s.split("---")[0] if "---" in s else s for s in style_names]
    parent_counts = Counter(parent_names)
    parents_sorted = sorted(parent_counts.items(), key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([p[0] for p in parents_sorted], [p[1] for p in parents_sorted], color=sns.color_palette("muted"))
    ax.set_xlabel("Genre")
    ax.set_ylabel("Track count")
    ax.set_title("Genre Distribution (broad parent genres)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(figures_dir / "genre_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved genre_distribution.png")


def plot_tempo_distribution(df: pd.DataFrame, figures_dir: Path):
    """Histogram of BPM values."""
    bpm = df["bpm"].dropna()
    if bpm.empty:
        print("  ⚠ No BPM data — skipping tempo plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(bpm, bins=60, range=(40, 220), color="steelblue", edgecolor="white")
    ax.set_xlabel("BPM")
    ax.set_ylabel("Track count")
    ax.set_title("Tempo Distribution")
    plt.tight_layout()
    fig.savefig(figures_dir / "tempo_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved tempo_distribution.png")


def plot_danceability_distribution(df: pd.DataFrame, figures_dir: Path):
    """Histogram of danceability probabilities."""
    dance = df["danceability"].dropna()
    if dance.empty:
        print("  ⚠ No danceability data — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dance, bins=40, range=(0, 1), color="coral", edgecolor="white")
    ax.set_xlabel("Danceability probability")
    ax.set_ylabel("Track count")
    ax.set_title("Danceability Distribution")
    plt.tight_layout()
    fig.savefig(figures_dir / "danceability_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved danceability_distribution.png")


def plot_key_scale_distribution(df: pd.DataFrame, figures_dir: Path):
    """Bar charts for key/scale from each of the 3 profiles, side by side."""
    profiles = ["temperley", "krumhansl", "edma"]
    available = [p for p in profiles if f"key_{p}" in df.columns and df[f"key_{p}"].notna().any()]
    if not available:
        print("  ⚠ No key data — skipping key plots.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), sharey=True)
    if len(available) == 1:
        axes = [axes]

    note_order = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    for ax, profile in zip(axes, available):
        key_col = f"key_{profile}"
        scale_col = f"scale_{profile}"
        sub = df[[key_col, scale_col]].dropna()
        combo = sub[key_col] + " " + sub[scale_col]
        counts = combo.value_counts()

        # Sort by note order
        sorted_labels = []
        sorted_vals = []
        for note in note_order:
            for scale in ("major", "minor"):
                label = f"{note} {scale}"
                sorted_labels.append(label)
                sorted_vals.append(counts.get(label, 0))

        colors = ["#4C72B0" if "major" in l else "#DD8452" for l in sorted_labels]
        ax.bar(range(len(sorted_labels)), sorted_vals, color=colors, edgecolor="white")
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
        ax.set_title(f"Key/Scale — {profile}")
        ax.set_ylabel("Track count")

    plt.tight_layout()
    fig.savefig(figures_dir / "key_scale_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved key_scale_distribution.png")


def plot_loudness_distribution(df: pd.DataFrame, figures_dir: Path):
    """Histogram of integrated loudness in LUFS with reference lines."""
    lufs = df["loudness_lufs"].dropna()
    if lufs.empty:
        print("  ⚠ No loudness data — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lufs, bins=50, color="mediumpurple", edgecolor="white")
    ax.axvline(-14, color="red", linestyle="--", linewidth=1.5, label="Streaming target (-14 LUFS)")
    ax.axvline(-23, color="orange", linestyle="--", linewidth=1.5, label="Broadcast ref (-23 LUFS)")
    ax.set_xlabel("Integrated Loudness (LUFS)")
    ax.set_ylabel("Track count")
    ax.set_title("Loudness Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "loudness_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved loudness_distribution.png")


def plot_voice_instrumental(df: pd.DataFrame, figures_dir: Path):
    """Pie chart of voice vs instrumental classification."""
    vp = df["voice_prob"].dropna()
    if vp.empty:
        print("  ⚠ No voice data — skipping plot.")
        return

    n_vocal = (vp >= 0.5).sum()
    n_instr = (vp < 0.5).sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [n_vocal, n_instr],
        labels=["Vocal", "Instrumental"],
        autopct="%1.1f%%",
        colors=["#66c2a5", "#fc8d62"],
        startangle=90,
    )
    ax.set_title("Voice vs Instrumental")
    plt.tight_layout()
    fig.savefig(figures_dir / "voice_instrumental.png", dpi=150)
    plt.close(fig)
    print("  Saved voice_instrumental.png")


def print_key_agreement(df: pd.DataFrame):
    """Print % of tracks where all 3 key profiles agree on key + scale."""
    profiles = ["temperley", "krumhansl", "edma"]
    cols_key = [f"key_{p}" for p in profiles]
    cols_scale = [f"scale_{p}" for p in profiles]

    if not all(c in df.columns for c in cols_key + cols_scale):
        print("  ⚠ Key columns missing — cannot compute agreement.")
        return

    sub = df[cols_key + cols_scale].dropna()
    if sub.empty:
        print("  ⚠ No complete key data for agreement check.")
        return

    agree = sum(
        1
        for _, row in sub.iterrows()
        if row[cols_key[0]] == row[cols_key[1]] == row[cols_key[2]]
        and row[cols_scale[0]] == row[cols_scale[1]] == row[cols_scale[2]]
    )
    pct = 100 * agree / len(sub)
    print(f"\n  Key-profile agreement (all 3 match): {agree}/{len(sub)} = {pct:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    """Load features.parquet and generate all report figures."""
    args = parse_args()
    results_dir = args.results_dir
    figures_dir = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = results_dir / "features.parquet"
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found. Run analyze_collection.py first.")
        return

    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} tracks loaded.\n")

    labels = load_discogs400_labels()

    print("Generating plots:")
    plot_genre_distribution(df, labels, figures_dir)
    plot_tempo_distribution(df, figures_dir)
    plot_danceability_distribution(df, figures_dir)
    plot_key_scale_distribution(df, figures_dir)
    plot_loudness_distribution(df, figures_dir)
    plot_voice_instrumental(df, figures_dir)
    print_key_agreement(df)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
