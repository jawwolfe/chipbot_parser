#!/usr/bin/env python3
"""
split_by_cluster.py

Reads a CSV with columns:
    file,start_time,end_time,birdnet_label,confidence,umap_x,umap_y,cluster

For every row, extracts the [start_time, end_time) segment from the
corresponding source WAV file, then concatenates all segments that share
the same `cluster` value (across all source files) into one output WAV
file per cluster.

Usage:
    python split_by_cluster.py \
        --csv segments.csv \
        --audio-dir /path/to/source/wavs \
        --out-dir /path/to/output \
        [--min-confidence 0.0] \
        [--gap-ms 0]

Notes:
- Segments for a given cluster are written in the order they appear in
  the CSV (grouped by cluster, but original file/time ordering within
  the cluster is preserved as found in the CSV).
- All source WAVs are assumed to share the same sample rate, sample
  width, and channel count (this is typical for recordings from the
  same device/deployment). If a mismatch is found, the script will
  raise a clear error rather than silently producing corrupt audio.
- `--gap-ms` optionally inserts N milliseconds of silence between
  concatenated segments in the output file (default: 0, i.e. segments
  are butted together with no gap).
- Rows are matched to files by exact filename in the `file` column;
  the script looks for that filename inside --audio-dir.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
import soundfile as sf  # Replaced wave with soundfile


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
    return p.parse_args()


def read_rows(csv_path, min_confidence):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"file", "start_time", "end_time", "cluster"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"CSV is missing required columns: {sorted(missing)}")
        for i, row in enumerate(reader, start=2):
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
                "label": row.get("birdnet_label", ""),
            })
    return rows


class SoundCache:
    """Caches SoundFile info objects to validate match structures."""

    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self._info = {}

    def get_info(self, filename):
        if filename not in self._info:
            path = os.path.join(self.audio_dir, filename)
            if not os.path.isfile(path):
                sys.exit(f"Source WAV not found: {path}")
            # sf.info reads metadata without loading the entire audio array into memory
            self._info[filename] = sf.info(path)
        return self._info[filename]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_rows(args.csv, args.min_confidence)
    if not rows:
        sys.exit("No usable rows found in CSV after filtering.")

    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster"]].append(row)

    cache = SoundCache(args.audio_dir)

    # Validate audio formats match
    ref_info = None
    ref_file = None
    all_files = sorted({row["file"] for row in rows})

    for fname in all_files:
        info = cache.get_info(fname)
        if ref_info is None:
            ref_info = info
            ref_file = fname
        else:
            if (info.samplerate, info.channels) != (ref_info.samplerate, ref_info.channels):
                sys.exit(
                    f"Format mismatch: '{fname}' ({info.samplerate}Hz, {info.channels}ch) differs from "
                    f"'{ref_file}' ({ref_info.samplerate}Hz, {ref_info.channels}ch)."
                )

    samplerate = ref_info.samplerate
    channels = ref_info.channels
    subtype = ref_info.subtype  # e.g., 'FLOAT' or 'PCM_16'

    import numpy as np

    # Create silence gaps using NumPy arrays (soundfile natively speaks numpy)
    gap_frames = int(round((args.gap_ms / 1000.0) * samplerate)) if args.gap_ms > 0 else 0
    gap_array = np.zeros((gap_frames, channels), dtype=np.float32) if gap_frames > 0 else None

    print(f"Found {len(rows)} segments across {len(all_files)} source file(s), {len(by_cluster)} cluster(s).")

    for cluster, group in sorted(by_cluster.items(), key=lambda kv: kv[0]):
        out_path = os.path.join(args.out_dir, f"{args.prefix}{cluster}.wav")
        total_duration = sum(r["end"] - r["start"] for r in group)
        print(f"  cluster {cluster}: {len(group)} segment(s), ~{total_duration:.1f}s -> {out_path}")

        # Open the output file using the source file's matching bit/float structure
        with sf.SoundFile(out_path, mode='w', samplerate=samplerate, channels=channels, subtype=subtype) as out_wf:
            for idx, row in enumerate(group):
                path = os.path.join(args.audio_dir, row["file"])
                info = cache.get_info(row["file"])

                # Calculate precise frame locations
                start_frame = max(0, int(round(row["start"] * info.samplerate)))
                end_frame = min(info.frames, int(round(row["end"] * info.samplerate)))
                frames_to_read = end_frame - start_frame

                if frames_to_read <= 0:
                    continue

                # Read only the target segment directly from disk
                with sf.SoundFile(path) as in_wf:
                    in_wf.seek(start_frame)
                    data = in_wf.read(frames_to_read)
                    out_wf.write(data)

                # Add gap if required
                if gap_array is not None and idx != len(group) - 1:
                    out_wf.write(gap_array)

    print("Done.")


if __name__ == "__main__":
    main()