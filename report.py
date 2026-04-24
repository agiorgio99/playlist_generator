#\!/usr/bin/env python3
"""
report.py - Generate an analysis report for a music collection.

Interactive: prompts for the collection to report on and the report name.
Saves to:  ./reports/<name>/report.html
           ./reports/<name>/figures/  (PNG plots)

Run:  python report.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── Config loading ─────────────────────────────────────────────────────────
def _load_yaml_config(config_path: Path = Path("config.yaml")) -> dict:
    defaults = {"results_base": "./analysis_results", "models_dir": "./models"}
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                loaded = yaml.safe_load(f) or {}
            defaults.update({k: v for k, v in loaded.items() if v is not None})
        except Exception as exc:
            print(f"Warning: could not read {config_path}: {exc}")
    return defaults


_yaml_cfg = _load_yaml_config()
RESULTS_BASE = Path(_yaml_cfg.get("results_base", "./analysis_results"))
MODELS_DIR   = Path(_yaml_cfg.get("models_dir",   "./models"))
REPORTS_DIR  = Path("./reports")


# ── Collection detection ───────────────────────────────────────────────────
def get_analyzed_collections(results_base: Path) -> list:
    if not results_base.exists():
        return []
    cols = []
    for d in sorted(results_base.iterdir()):
        if not d.is_dir():
            continue
        parquet = d / "features.parquet"
        if parquet.exists() and parquet.stat().st_size > 0:
            info = {}
            info_file = d / "collection_info.json"
            if info_file.exists():
                try:
                    info = json.loads(info_file.read_text())
                except Exception:
                    pass
            cols.append({"name": d.name, "parquet": parquet, "info": info})
    return cols


# ── Interactive setup ──────────────────────────────────────────────────────
def interactive_setup(args) -> tuple:
    """
    Returns (parquet_path, report_dir, models_dir).
    Uses CLI args if provided; otherwise prompts interactively.
    """
    collections = get_analyzed_collections(RESULTS_BASE)

    if not collections:
        print(
            "\n  No analyzed collections found under ./analysis_results/\n"
            "  Run 'python analyze_collection.py' first.\n"
        )
        sys.exit(1)

    # ── Pick collection ──────────────────────────────────────────────────
    if args.collection:
        match = next((c for c in collections if c["name"] == args.collection), None)
        if match is None:
            print(f"  Collection '{args.collection}' not found.")
            sys.exit(1)
        chosen = match
    else:
        print()
        print("=" * 62)
        print("  AVAILABLE COLLECTIONS")
        print("=" * 62)
        for i, c in enumerate(collections, 1):
            analyzed_at = c["info"].get("analyzed_at", "unknown date")
            try:
                df_tmp = pd.read_parquet(c["parquet"], columns=["track_id"])
                n = len(df_tmp)
            except Exception:
                n = "?"
            print(f"  [{i}]  {c['name']}  ({n} tracks, analyzed {analyzed_at})")
        print()
        raw = input(f"  Select collection [1–{len(collections)}]: ").strip()
        try:
            idx = int(raw) - 1
            if not (0 <= idx < len(collections)):
                raise ValueError
        except ValueError:
            print("  Invalid selection.")
            sys.exit(1)
        chosen = collections[idx]

    parquet_path = chosen["parquet"]
    collection_name = chosen["name"]
    print(f"\n  Collection: {collection_name}")

    # ── Report name ──────────────────────────────────────────────────────
    if args.report_name:
        report_name = args.report_name
    else:
        default_name = f"report_{collection_name}"
        raw = input(f"  Report name [{default_name}]: ").strip()
        report_name = raw if raw else default_name

    report_dir = REPORTS_DIR / report_name
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output:     {report_dir}\n")

    models_dir = args.models_dir or MODELS_DIR
    return parquet_path, report_dir, figures_dir, Path(models_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Generate a music collection report.")
    p.add_argument("--collection",  type=str,  default=None,
                   help="Collection name (subfolder of analysis_results/). Skips prompt.")
    p.add_argument("--report-name", type=str,  default=None,
                   help="Report folder name under reports/. Skips prompt.")
    p.add_argument("--models-dir",  type=Path, default=None,
                   help="Folder containing Essentia model JSON for genre labels.")
    return p.parse_args()


# ── Label loading ──────────────────────────────────────────────────────────
def load_discogs400_labels(models_dir: Path) -> list:
    label_file = models_dir / "genre_discogs400-discogs-effnet-1.json"
    if label_file.exists():
        with open(label_file) as f:
            return json.load(f).get("classes", [f"class_{i}" for i in range(400)])
    txt_file = models_dir / "discogs400_labels.txt"
    if txt_file.exists():
        return [l.strip() for l in txt_file.read_text().splitlines() if l.strip()]
    return [f"class_{i}" for i in range(400)]


# ── Plot functions ─────────────────────────────────────────────────────────
def plot_genre_distribution(df, labels, figures_dir, top_n=20):
    if "genre_discogs400" not in df.columns or df["genre_discogs400"].isna().all():
        print("  ⚠ genre_discogs400 missing — skipping.")
        return
    genre_vecs  = np.stack(df["genre_discogs400"].dropna().values)
    top_indices = np.argmax(genre_vecs, axis=1)
    style_names = [labels[i] for i in top_indices] if len(labels) == genre_vecs.shape[1] \
                  else [f"class_{i}" for i in top_indices]

    style_counts = Counter(style_names)
    style_df = pd.DataFrame(
        sorted(style_counts.items(), key=lambda x: -x[1]), columns=["style", "count"]
    )
    style_df.to_csv(figures_dir / "styles_distribution.tsv", sep="\t", index=False)
    print(f"  Saved styles_distribution.tsv ({len(style_df)} unique styles)")

    parent_names  = [s.split("---")[0] if "---" in s else s for s in style_names]
    parent_counts = Counter(parent_names)
    parents       = list(reversed(sorted(parent_counts.items(), key=lambda x: -x[1])[:top_n]))

    fig, ax = plt.subplots(figsize=(10, max(4, len(parents) * 0.35)))
    ax.barh([p[0] for p in parents], [p[1] for p in parents],
            color=sns.color_palette("muted")[0])
    ax.set_xlabel("Track count")
    ax.set_title(f"Genre Distribution (top {top_n} broad parent genres)")
    plt.tight_layout()
    fig.savefig(figures_dir / "genre_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved genre_distribution.png")


def plot_tempo_distribution(df, figures_dir):
    bpm = df["bpm"].dropna()
    if bpm.empty:
        print("  ⚠ No BPM data — skipping.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(bpm, bins=60, range=(40, 220), color="steelblue", edgecolor="white")
    ax.set_xlabel("BPM"); ax.set_ylabel("Track count"); ax.set_title("Tempo Distribution")
    plt.tight_layout()
    fig.savefig(figures_dir / "tempo_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved tempo_distribution.png")


def plot_danceability_distribution(df, figures_dir):
    dance = df["danceability"].dropna()
    if dance.empty:
        print("  ⚠ No danceability data — skipping.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dance, bins=40, range=(0, 1), color="coral", edgecolor="white")
    ax.set_xlabel("Danceability probability"); ax.set_ylabel("Track count")
    ax.set_title("Danceability Distribution")
    plt.tight_layout()
    fig.savefig(figures_dir / "danceability_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved danceability_distribution.png")


def plot_key_scale_distribution(df, figures_dir):
    profiles  = ["temperley", "krumhansl", "edma"]
    available = [p for p in profiles if f"key_{p}" in df.columns and df[f"key_{p}"].notna().any()]
    if not available:
        print("  ⚠ No key data — skipping.")
        return
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), sharey=True)
    if len(available) == 1:
        axes = [axes]
    note_order = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for ax, profile in zip(axes, available):
        sub   = df[[f"key_{profile}", f"scale_{profile}"]].dropna()
        combo = sub[f"key_{profile}"] + " " + sub[f"scale_{profile}"]
        counts = combo.value_counts()
        labels_sorted = []
        vals = []
        for note in note_order:
            for scale in ("major", "minor"):
                lbl = f"{note} {scale}"
                labels_sorted.append(lbl)
                vals.append(counts.get(lbl, 0))
        colors = ["#4C72B0" if "major" in l else "#DD8452" for l in labels_sorted]
        ax.bar(range(len(labels_sorted)), vals, color=colors, edgecolor="white")
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, rotation=90, fontsize=7)
        ax.set_title(f"Key/Scale — {profile}")
        ax.set_ylabel("Track count")
    plt.tight_layout()
    fig.savefig(figures_dir / "key_scale_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved key_scale_distribution.png")


def plot_loudness_distribution(df, figures_dir):
    lufs = df["loudness_lufs"].dropna()
    if lufs.empty:
        print("  ⚠ No loudness data — skipping.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lufs, bins=50, color="mediumpurple", edgecolor="white")
    ax.axvline(-14, color="red",    linestyle="--", linewidth=1.5, label="Streaming target (-14 LUFS)")
    ax.axvline(-23, color="orange", linestyle="--", linewidth=1.5, label="Broadcast ref (-23 LUFS)")
    ax.set_xlabel("Integrated Loudness (LUFS)"); ax.set_ylabel("Track count")
    ax.set_title("Loudness Distribution"); ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "loudness_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved loudness_distribution.png")


def plot_voice_instrumental(df, figures_dir):
    vp = df["voice_prob"].dropna()
    if vp.empty:
        print("  ⚠ No voice data — skipping.")
        return
    n_vocal = (vp >= 0.5).sum()
    n_instr = (vp < 0.5).sum()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([n_vocal, n_instr], labels=["Vocal", "Instrumental"], autopct="%1.1f%%",
           colors=["#66c2a5", "#fc8d62"], startangle=90)
    ax.set_title("Voice vs Instrumental")
    plt.tight_layout()
    fig.savefig(figures_dir / "voice_instrumental.png", dpi=150)
    plt.close(fig)
    print("  Saved voice_instrumental.png")


# ── HTML report ────────────────────────────────────────────────────────────
def generate_html_report(df, figures_dir, report_dir, collection_name):
    figure_files = [
        ("genre_distribution.png",      "Genre Distribution"),
        ("tempo_distribution.png",      "Tempo Distribution"),
        ("danceability_distribution.png","Danceability Distribution"),
        ("key_scale_distribution.png",  "Key / Scale Distribution"),
        ("loudness_distribution.png",   "Loudness Distribution"),
        ("voice_instrumental.png",      "Voice vs Instrumental"),
    ]

    stats_rows = ""
    for col, label in [
        ("bpm",          "BPM (mean ± std)"),
        ("loudness_lufs","Loudness LUFS (mean ± std)"),
        ("danceability", "Danceability (mean ± std)"),
        ("voice_prob",   "Voice prob (mean ± std)"),
    ]:
        if col in df.columns and df[col].notna().any():
            m, s = df[col].mean(), df[col].std()
            stats_rows += f"<tr><td>{label}</td><td>{m:.3f} ± {s:.3f}</td></tr>\n"

    profiles   = ["temperley", "krumhansl", "edma"]
    cols_key   = [f"key_{p}"   for p in profiles]
    cols_scale = [f"scale_{p}" for p in profiles]
    agreement_html = ""
    if all(c in df.columns for c in cols_key + cols_scale):
        sub = df[cols_key + cols_scale].dropna()
        if not sub.empty:
            agree = sum(
                1 for _, row in sub.iterrows()
                if row[cols_key[0]] == row[cols_key[1]] == row[cols_key[2]]
                and row[cols_scale[0]] == row[cols_scale[1]] == row[cols_scale[2]]
            )
            pct = 100 * agree / len(sub)
            agreement_html = (
                f"<p><strong>Key-profile agreement (all 3 match):</strong> "
                f"{agree}/{len(sub)} = {pct:.1f}%</p>"
            )

    # Figures referenced by relative path (figures/ subfolder)
    figures_html = ""
    for fname, title in figure_files:
        if (figures_dir / fname).exists():
            figures_html += f"""
    <section>
      <h2>{title}</h2>
      <img src="figures/{fname}" alt="{title}"
           style="max-width:100%;border:1px solid #ddd;border-radius:4px;">
    </section>
"""

    html = f"""<\!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Report — {collection_name}</title>
  <style>
    body  {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1    {{ border-bottom: 2px solid #4C72B0; padding-bottom: 8px; }}
    h2    {{ color: #4C72B0; margin-top: 40px; }}
    table {{ border-collapse: collapse; margin: 12px 0; }}
    td, th {{ border: 1px solid #ccc; padding: 6px 14px; }}
    th    {{ background: #f0f4fb; }}
    section {{ margin-bottom: 40px; }}
  </style>
</head>
<body>
  <h1>Music Collection Analysis Report</h1>
  <p><strong>Collection:</strong> {collection_name}</p>
  <p><strong>Tracks analysed:</strong> {len(df)}</p>

  <h2>Summary Statistics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {stats_rows}
  </table>
  {agreement_html}

  {figures_html}
</body>
</html>
"""
    out = report_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Saved report.html ({out.stat().st_size // 1024} KB)")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    parquet_path, report_dir, figures_dir, models_dir = interactive_setup(args)

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} tracks from features.parquet\n")

    labels = load_discogs400_labels(models_dir)

    print("Generating figures …")
    plot_genre_distribution(df, labels, figures_dir)
    plot_tempo_distribution(df, figures_dir)
    plot_danceability_distribution(df, figures_dir)
    plot_key_scale_distribution(df, figures_dir)
    plot_loudness_distribution(df, figures_dir)
    plot_voice_instrumental(df, figures_dir)

    print("\nGenerating HTML report …")
    generate_html_report(df, figures_dir, report_dir, chosen_name := report_dir.name)
    print(f"\nDone.  Open: {report_dir / 'report.html'}")


if __name__ == "__main__":
    main()
