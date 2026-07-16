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

Usage:
    python split_by_cluster_identified.py \
        --csv segments.csv \
        --audio-dir /path/to/source/wavs \
        --out-dir /path/to/output \
        [--min-confidence 0.0] \
        [--gap-ms 0]

Notes:
- "Identified species" means birdnet_label is present and not equal to
  "Unidentified/Ambient" (case-insensitive, whitespace-trimmed match).
- All source WAVs are assumed to share the same sample rate, sample
  width, and channel count. If a mismatch is found, the script raises a
  clear error rather than silently producing corrupt audio.
- `--gap-ms` optionally inserts N milliseconds of silence between
  concatenated segments in the output file (default: 0).
"""

import argparse
import csv
import os
import re
import sys
import wave
from collections import defaultdict

IGNORED_LABEL = "unidentified/ambient"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Path to the input CSV file")
    p.add_argument("--audio-dir", required=True, help="Directory containing the source WAV files")
    p.add_argument("--out-dir", required=True, help="Directory to write per-cluster output WAV files")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="Optional: skip rows with confidence below this value")
    p.add_argument("--gap-ms", type=float, default=0.0,
                   help="Silence (ms) to insert between concatenated segments (default: 0)")
    p.add_argument("--prefix", default="cluster_", help="Filename prefix for outputs (default: 'cluster_')")
    p.add_argument("--max-species-name-length", type=int, default=120,
                   help="Max characters for the species portion of the filename before it's "
                        "truncated with a '+N_more' suffix (default: 120)")
    return p.parse_args()


def read_rows(csv_path, min_confidence):
    rows = []
    skipped_unidentified = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"file", "start_time", "end_time", "cluster", "birdnet_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"CSV is missing required columns: {sorted(missing)}")
        for i, row in enumerate(reader, start=2):  # header is line 1
            label = (row.get("birdnet_label") or "").strip()
            if not label or label.lower() == IGNORED_LABEL:
                skipped_unidentified += 1
                continue
            try:
                start = float(row["start_time"])
                end = float(row["end_time"])
                cluster = row["cluster"].strip()
                conf = float(row["confidence"]) if row.get("confidence") not in (None, "") else None
            except ValueError as e:
                sys.exit(f"CSV row {i}: could not parse numeric field ({e})")
            if end <= start:
                print(f"Warning: row {i} has end_time <= start_time, skipping", file=sys.stderr)
                continue
            if min_confidence is not None and conf is not None and conf < min_confidence:
                continue
            rows.append({
                "file": row["file"].strip(),
                "start": start,
                "end": end,
                "cluster": cluster,
                "confidence": conf,
                "label": label,
            })
    print(f"Skipped {skipped_unidentified} row(s) labeled '{IGNORED_LABEL}' or blank.")
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
    # BirdNET labels are sometimes "Scientific name_Common Name" - keep the
    # common name (last underscore-separated part) if that pattern is present.
    if "_" in cleaned and cleaned.count("_") == 1:
        _, _, common = cleaned.partition("_")
        if common:
            cleaned = common
    cleaned = cleaned.replace("/", "-").replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned or "unknown_species"


def all_species_slug(group, max_length=120):
    """Return a filename-safe chunk listing distinct species in a cluster,
    sorted alphabetically and joined with '+'. If the full list would make
    the filename too long (a real risk with big/mixed "noise" clusters,
    especially on Windows where full paths are capped around 260 chars),
    it's truncated and a "+N_more" suffix is appended instead of silently
    producing a file the OS then refuses to create.
    """
    slugs = sorted({sanitize_for_filename(r["label"]) for r in group})
    full = "+".join(slugs)
    if len(full) <= max_length:
        return full

    kept = []
    length = 0
    for slug in slugs:
        added = (1 if kept else 0) + len(slug)  # +1 for the joining "+"
        if length + added > max_length:
            break
        kept.append(slug)
        length += added
    remaining = len(slugs) - len(kept)
    if not kept:
        # even a single species name is too long on its own - hard-truncate it
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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_rows(args.csv, args.min_confidence)
    if not rows:
        sys.exit("No identified-species rows found in CSV after filtering.")

    # Group rows by cluster, then sort each group by filename, then start_time
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster"]].append(row)
    for cluster in by_cluster:
        by_cluster[cluster].sort(key=lambda r: (r["file"], r["start"]))

    cache = WavCache(args.audio_dir)

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
                    f"Format mismatch: '{fname}' ({params.framerate}Hz, "
                    f"{params.sampwidth*8}-bit, {params.nchannels}ch) differs from "
                    f"'{reference_file}' ({reference_params.framerate}Hz, "
                    f"{reference_params.sampwidth*8}-bit, {reference_params.nchannels}ch). "
                    f"Resample/convert sources to a common format before running this script."
                )

    framerate = reference_params.framerate
    sampwidth = reference_params.sampwidth
    nchannels = reference_params.nchannels
    silence_frame = b"\x00" * (sampwidth * nchannels)
    gap_frames = int(round((args.gap_ms / 1000.0) * framerate)) if args.gap_ms > 0 else 0
    gap_bytes = silence_frame * gap_frames

    print(f"Found {len(rows)} identified-species segments across {len(all_files)} "
          f"source file(s), {len(by_cluster)} cluster(s).")

    for cluster, group in sorted(by_cluster.items(), key=lambda kv: kv[0]):
        labels = sorted({r["label"] for r in group})
        species_slug = all_species_slug(group, max_length=args.max_species_name_length)
        out_path = os.path.join(args.out_dir, f"{args.prefix}{cluster}_{species_slug}.wav")
        total_duration = sum(r["end"] - r["start"] for r in group)
        print(f"  cluster {cluster}: {len(group)} segment(s), "
              f"~{total_duration:.1f}s, species: {', '.join(labels)} -> {out_path}")
        if len(os.path.abspath(out_path)) > 245:
            print(f"    Warning: full output path is {len(os.path.abspath(out_path))} chars long; "
                  f"this may fail on Windows (260-char limit). Consider --max-species-name-length "
                  f"with a smaller value, or a shorter --out-dir.", file=sys.stderr)

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

    cache.close_all()
    print("Done.")


if __name__ == "__main__":
    main()