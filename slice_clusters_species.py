#!/usr/bin/env python3
"""
split_by_cluster_identified.py

Same as split_by_cluster.py, but:
  - Skips any row whose birdnet_label is "Unidentified/Ambient" (i.e. only
    keeps rows that BirdNET identified as an actual species).
  - Within each cluster, orders the extracted segments by filename, then
    by start_time, rather than by CSV row order.
  - Names each output file after its cluster number AND every distinct
    species label found in that cluster, sorted alphabetically and joined
    with "+", e.g. "cluster_5_American_Robin+Blue_Jay.wav".

Reads a CSV with columns:
    file,start_time,end_time,birdnet_label,confidence,umap_x,umap_y,cluster

For every remaining row, extracts the [start_time, end_time) segment from
the corresponding source WAV file, then concatenates all segments that
share the same `cluster` value (across all source files, sorted by
filename then start_time) into one output WAV file per cluster.

Notes:
- "Identified species" means birdnet_label is present and not equal to
  "Unidentified/Ambient" (case-insensitive, whitespace-trimmed match).
- All source WAVs are assumed to share the same sample rate, sample
  width, and channel count. If a mismatch is found, the script raises a
  clear error rather than silently producing corrupt audio.
"""

import csv
import os
import re
import sys
import wave
from collections import defaultdict
from pathlib import Path

# We lowercase everything to check matching safely
IGNORED_LABEL = "unidentified/ambient"


def read_rows(csv_path, min_confidence):
    rows = []
    skipped_blank = 0
    skipped_low_confidence = 0

    with open(csv_path, newline="") as f:
        # Custom dict reader wrapper to handle case-insensitive headers just in case
        raw_reader = csv.DictReader(f)
        if not raw_reader.fieldnames:
            sys.exit("CSV appears to be empty or missing a header row.")

        # Map headers to lowercase to prevent "birdnet_label" vs "BirdNet_label" mismatches
        header_map = {name.lower().strip(): name for name in raw_reader.fieldnames}

        required = {"file", "start_time", "end_time", "cluster", "birdnet_label"}
        missing = required - set(header_map.keys())
        if missing:
            sys.exit(f"CSV is missing required columns: {sorted(missing)}")

        for i, raw_row in enumerate(raw_reader, start=2):  # header is line 1
            # Reconstruct row using lowercase keys for safe access
            row = {k.lower().strip(): v for k, v in raw_row.items() if k}

            label = (row.get("birdnet_label") or "").strip()
            if not label:
                skipped_blank += 1
                continue

            try:
                start = float(row["start_time"])
                end = float(row["end_time"])
                cluster = row["cluster"].strip()
                # Safely get confidence if it exists
                conf_val = row.get("confidence")
                conf = float(conf_val) if conf_val not in (None, "") else None
            except ValueError as e:
                sys.exit(f"CSV row {i}: could not parse numeric field ({e})")

            if end <= start:
                print(f"Warning: row {i} has end_time <= start_time, skipping", file=sys.stderr)
                continue

            # Determine if this row belongs to the ambient/unidentified category
            is_ambient = label.lower() == IGNORED_LABEL

            # CRITICAL FIX: Only apply min_confidence to IDENTIFIED species.
            # Ambient/Unidentified clips often have 0.0 or low confidence scores.
            if not is_ambient and min_confidence is not None and conf is not None and conf < min_confidence:
                skipped_low_confidence += 1
                continue

            rows.append({
                "file": row["file"].strip(),
                "start": start,
                "end": end,
                "cluster": cluster,
                "confidence": conf,
                "label": label,  # Keeps original casing for filename
                "is_ambient": is_ambient
            })

    print(f"Loaded {len(rows)} total valid rows.")
    print(f"  - Skipped {skipped_blank} blank label rows.")
    print(f"  - Skipped {skipped_low_confidence} identified species rows below {min_confidence} confidence.")
    return rows


class WavCache:
    """Keeps source WAV files open (read-only) and caches their audio params."""

    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self._handles = {}
        self._params = {}

    def get(self, filename):
        if filename not in self._handles:
            path = os.path.join(self.audio_dir, filename)
            if not os.path.isfile(path):
                sys.exit(f"Source WAV not found: {path}")
            wf = wave.open(path, "rb")
            self._handles[filename] = wf
            self._params[filename] = wf.getparams()
        return self._handles[filename]

    def params(self, filename):
        self.get(filename)
        return self._params[filename]

    def close_all(self):
        for wf in self._handles.values():
            wf.close()


def sanitize_for_filename(label):
    """Turn a birdnet_label into a filesystem-safe chunk for use in a filename."""
    cleaned = label.strip()
    if "_" in cleaned and cleaned.count("_") == 1:
        _, _, common = cleaned.partition("_")
        if common:
            cleaned = common
    cleaned = cleaned.replace("/", "-").replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned or "unknown_species"


def all_species_slug(group, max_length=120):
    """Return a filename-safe chunk listing distinct species in a cluster."""
    slugs = sorted({sanitize_for_filename(r["label"]) for r in group})
    full = "+".join(slugs)
    if len(full) <= max_length:
        return full

    kept = []
    length = 0
    for slug in slugs:
        added = (1 if kept else 0) + len(slug)
        if length + added > max_length:
            break
        kept.append(slug)
        length += added
    remaining = len(slugs) - len(kept)
    if not kept:
        return slugs[0][:max_length]
    return "+".join(kept) + f"+{remaining}_more"


def extract_segment_bytes(wf, params, start_time, end_time):
    framerate = params.framerate
    n_frames_total = params.nframes
    sampwidth = params.sampwidth
    nchannels = params.nchannels

    start_frame = max(0, int(round(start_time * framerate)))
    end_frame = min(n_frames_total, int(round(end_time * framerate)))
    if start_frame >= end_frame:
        return b""

    wf.setpos(start_frame)
    n_frames = end_frame - start_frame
    data = wf.readframes(n_frames)
    expected_bytes = n_frames * sampwidth * nchannels
    if len(data) < expected_bytes:
        print(f"Warning: requested {expected_bytes} bytes but only got {len(data)} "
              f"(segment near end of file, truncated)", file=sys.stderr)
    return data


def write_cluster_wavs(by_cluster, cache, destination_dir, file_prefix, max_species_name_length, audio_format_params,
                       gap_bytes):
    """Helper to process a dictionary of clusters and write out combined WAV files."""
    nchannels, sampwidth, framerate = audio_format_params

    for cluster, group in sorted(by_cluster.items(), key=lambda kv: kv[0]):
        labels = sorted({r["label"] for r in group})
        species_slug = all_species_slug(group, max_length=max_species_name_length)
        out_path = os.path.join(destination_dir, f"{file_prefix}{cluster}_{species_slug}.wav")
        total_duration = sum(r["end"] - r["start"] for r in group)

        print(f"  cluster {cluster}: {len(group)} segment(s), "
              f"~{total_duration:.1f}s, labels: {', '.join(labels)} -> {out_path}")

        if len(os.path.abspath(out_path)) > 245:
            print(f"    Warning: full output path is {len(os.path.abspath(out_path))} chars long; "
                  f"this may fail on Windows (260-char limit).", file=sys.stderr)

        with wave.open(out_path, "wb") as out_wf:
            out_wf.setnchannels(nchannels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)

            for idx, row in enumerate(group):
                wf = cache.get(row["file"])
                params = cache.params(row["file"])
                data = extract_segment_bytes(wf, params, row["start"], row["end"])
                out_wf.writeframes(data)
                if gap_bytes and idx != len(group) - 1:
                    out_wf.writeframes(gap_bytes)


def run(csv_in, audio_dir, out_dir_species, out_dir_ambient, min_confidence=None, gap_ms=0.0, file_prefix="cluster_",
        max_species_name_length=120):
    os.makedirs(out_dir_species, exist_ok=True)
    os.makedirs(out_dir_ambient, exist_ok=True)

    rows = read_rows(csv_in, min_confidence)
    if not rows:
        sys.exit("No matching rows found in CSV after filtering.")

    # Split rows into species vs ambient datasets
    species_rows = [r for r in rows if not r["is_ambient"]]
    ambient_rows = [r for r in rows if r["is_ambient"]]

    # Group species rows by cluster
    by_cluster_species = defaultdict(list)
    for row in species_rows:
        by_cluster_species[row["cluster"]].append(row)
    for cluster in by_cluster_species:
        by_cluster_species[cluster].sort(key=lambda r: (r["file"], r["start"]))

    # Group ambient rows by cluster
    by_cluster_ambient = defaultdict(list)
    for row in ambient_rows:
        by_cluster_ambient[row["cluster"]].append(row)
    for cluster in by_cluster_ambient:
        by_cluster_ambient[cluster].sort(key=lambda r: (r["file"], r["start"]))

    cache = WavCache(audio_dir)

    # Validate all referenced files share the same audio format up front
    reference_params = None
    reference_file = None
    all_files = sorted({row["file"] for row in rows})
    for fname in all_files:
        params = cache.params(fname)
        if reference_params is None:
            reference_params = params
            reference_file = fname
        else:
            if (params.framerate, params.sampwidth, params.nchannels) != \
                    (reference_params.framerate, reference_params.sampwidth, reference_params.nchannels):
                sys.exit(
                    f"Format mismatch: '{fname}' differs from '{reference_file}' format."
                )

    framerate = reference_params.framerate
    sampwidth = reference_params.sampwidth
    nchannels = reference_params.nchannels
    silence_frame = b"\x00" * (sampwidth * nchannels)
    gap_frames = int(round((gap_ms / 1000.0) * framerate)) if gap_ms > 0 else 0
    gap_bytes = silence_frame * gap_frames
    audio_format_params = (nchannels, sampwidth, framerate)

    # Processing Identified Species
    if by_cluster_species:
        print(
            f"\nProcessing {len(species_rows)} identified-species segments across {len(by_cluster_species)} cluster(s)...")
        write_cluster_wavs(by_cluster_species, cache, out_dir_species, file_prefix, max_species_name_length,
                           audio_format_params, gap_bytes)
    else:
        print("\nNo identified-species segments found to export.")

    # Processing Unidentified / Ambient
    if by_cluster_ambient:
        print(
            f"\nProcessing {len(ambient_rows)} unidentified/ambient segments across {len(by_cluster_ambient)} cluster(s)...")
        write_cluster_wavs(by_cluster_ambient, cache, out_dir_ambient, file_prefix, max_species_name_length,
                           audio_format_params, gap_bytes)
    else:
        print("\nNo unidentified/ambient segments found to export.")

    cache.close_all()
    print("\nDone.")


if __name__ == "__main__":
    ROOT_PATH = Path(r"C:\temp\CHIPBOT_DATA_ROOT")
    BATCH_NAME = "aw_chipbot_01_2026-07-17_20_06_24_39.875578_-86.283721"
    ANALYSIS_TIMESTAMP = "20260718_130151"

    run_folder_name = f"{BATCH_NAME}_{ANALYSIS_TIMESTAMP}"
    output_base = ROOT_PATH / "output" / run_folder_name

    CLUSTERS_CSV = output_base / f"acoustic_clusters_{run_folder_name}.csv"
    AUDIO_DIR = ROOT_PATH / "input" / "processed" / BATCH_NAME

    OUT_DIR_SPECIES = output_base / "species"
    OUT_DIR_CLUSTERS = output_base / "clusters"

    MIN_CONFIDENCE = .70
    GAP_MS = 0.0
    FILE_PREFIX = "cluster_"
    MAX_SPECIES_NAME_LENGTH = 120

    run(csv_in=CLUSTERS_CSV, audio_dir=AUDIO_DIR,
        out_dir_species=OUT_DIR_SPECIES, out_dir_ambient=OUT_DIR_CLUSTERS,
        min_confidence=MIN_CONFIDENCE, gap_ms=GAP_MS, file_prefix=FILE_PREFIX,
        max_species_name_length=MAX_SPECIES_NAME_LENGTH)