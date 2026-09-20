#!/usr/bin/env python3
"""
Rename carved cluster WAV clips from a 3-column list and re-save them with gain.

Input list: 3 columns -> cluster, segment, desc
  (tab, comma, or whitespace separated; '#' comments and an optional header row
   starting with "cluster" are ignored)

File pattern:      cluster_6_segment_1769_1930_100_-22.wav
  cluster value =  token after "cluster_"   -> 6
  segment value =  token after "segment_"   -> 1769

Rename rule: replace everything after the 2nd underscore from the END with desc
  cluster_6_segment_1769_1930_100_-22.wav  +  desc="Barred Owl"
  -> cluster_6_segment_1769_1930_Barred_Owl.wav

Gain: GAIN_DB (decibels) is applied when the file is written.

Usage:
  python rename_cluster_clips.py list.txt --root /path/to/runs --out ./renamed
  python rename_cluster_clips.py list.txt --root . --gain 6 --dry-run
  python rename_cluster_clips.py list.txt --root . --in-place
"""

import argparse
import array
import math
import re
import shutil
import sys
import wave
from pathlib import Path

# ---- defaults you can edit ------------------------------------------------
GAIN_DB = 25.0          # gain applied to every written file, in dB

SEARCH_ROOT = Path("/mnt/c/") / Path("temp/CHIPBOT_DATA_ROOT") / Path("links_clusters/run_1")
OUTPUT_DIR = r"/mnt/c/temp/CHIPBOT_DATA_ROOT/collected_clusters"
LIST_FILE = r"/mnt/c/Users/Andrew Wolfe/PycharmProjectsP/chip_bot_parser/clusters_collect.txt"
IN_PLACE    = False
DRY_RUN     = False
# ---------------------------------------------------------------------------

try:
    import numpy as np
    import soundfile as sf
    HAVE_SF = True
except Exception:
    HAVE_SF = False


# ------------------------- list file ---------------------------------------

def read_list(path):
    """Return [(cluster, segment, desc), ...]."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t")]
            elif "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split(None, 2)
            parts = [p for p in parts if p != ""]
            if len(parts) < 3:
                print(f"  ! line {lineno}: need 3 columns, got {len(parts)} -- skipped")
                continue
            cluster, segment, desc = parts[0], parts[1], " ".join(parts[2:])
            if lineno == 1 and cluster.lower() == "cluster" and segment.lower() == "segment":
                continue  # header
            rows.append((cluster, segment, desc))
    return rows


def norm(value):
    """Normalize a key so '06' == '6' and case doesn't matter."""
    v = str(value).strip()
    if v.lstrip("+-").isdigit():
        return str(int(v))
    return v.lower()


# ------------------------- filename parsing --------------------------------

def parse_stem(stem):
    """Pull cluster/segment values out of cluster_6_segment_1769_1930_100_-22."""
    parts = stem.split("_")
    found = {}
    for i, tok in enumerate(parts[:-1]):
        low = tok.lower()
        if low in ("cluster", "segment") and low not in found:
            found[low] = parts[i + 1]
    # positional fallback if the labels aren't there
    if "cluster" not in found and len(parts) > 1:
        found["cluster"] = parts[1]
    if "segment" not in found and len(parts) > 3:
        found["segment"] = parts[3]
    return found.get("cluster"), found.get("segment")


def safe_desc(desc):
    """Make desc safe for a filename."""
    d = desc.strip().replace(" ", "_")
    d = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", d)
    return d.strip("._") or "unnamed"


def new_stem(stem, desc):
    """Drop the last two underscore-separated tokens, append desc."""
    head = stem.rsplit("_", 2)[0]
    return f"{head}_{safe_desc(desc)}"


def find_wavs(root):
    """WAV files in root and in its immediate subdirectories (one level deep)."""
    root = Path(root)
    files = []
    for pattern in ("*.wav", "*.WAV", "*/*.wav", "*/*.WAV"):
        files.extend(root.glob(pattern))
    return sorted(set(files))


# ------------------------- audio -------------------------------------------

def write_with_gain(src, dst, gain_db):
    """Copy src -> dst applying gain_db. Returns True if any sample clipped."""
    factor = 10.0 ** (gain_db / 20.0)
    if HAVE_SF:
        return _write_sf(src, dst, factor)
    return _write_wave(src, dst, factor)


def _write_sf(src, dst, factor):
    info = sf.info(str(src))
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    data = data * factor
    clipped = bool(np.any(np.abs(data) > 1.0))
    np.clip(data, -1.0, 1.0, out=data)
    subtype = info.subtype if sf.check_format(info.format, info.subtype) else None
    sf.write(str(dst), data, sr, subtype=subtype, format="WAV")
    return clipped


_TYPECODE = {1: "b", 2: "h", 4: "i"}  # sample width -> array typecode


def _write_wave(src, dst, factor):
    """stdlib fallback: PCM 8/16/32-bit only."""
    with wave.open(str(src), "rb") as w:
        nch, width, sr, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nframes)
    if width not in _TYPECODE:
        raise RuntimeError(f"{width * 8}-bit WAV needs soundfile (pip install soundfile numpy)")

    samples = array.array(_TYPECODE[width])
    samples.frombytes(raw)
    if sys.byteorder == "big":
        samples.byteswap()

    offset = 128 if width == 1 else 0          # 8-bit WAV is unsigned
    lo, hi = -(2 ** (width * 8 - 1)), 2 ** (width * 8 - 1) - 1
    clipped = False
    for i, s in enumerate(samples):
        v = int(round((s - offset if width == 1 else s) * factor))
        if v > hi:
            v, clipped = hi, True
        elif v < lo:
            v, clipped = lo, True
        samples[i] = v + offset if width == 1 else v

    if sys.byteorder == "big":
        samples.byteswap()
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return clipped


# ------------------------- main --------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Rename carved cluster clips and apply gain.")
    ap.add_argument("list_file", nargs="?", default=LIST_FILE, help="text file: cluster, segment, desc")
    ap.add_argument("--root", default=SEARCH_ROOT, help="directory to search (one level deep)")
    ap.add_argument("--out", default=OUTPUT_DIR, help="output directory (ignored with --in-place)")
    ap.add_argument("--gain", type=float, default=GAIN_DB, help="gain in dB (default %(default)s)")
    ap.add_argument("--in-place", action="store_true", default=IN_PLACE,
                    help="overwrite the original file and rename it instead of writing to --out")
    ap.add_argument("--dry-run", action="store_true", default=DRY_RUN,
                    help="show what would happen, write nothing")
    args = ap.parse_args()

    rows = read_list(args.list_file)
    if not rows:
        sys.exit("No usable rows in the list file.")

    wavs = find_wavs(args.root)
    if not wavs:
        sys.exit(f"No WAV files found in {args.root} or its subdirectories.")

    # index every wav by (cluster, segment)
    index = {}
    for p in wavs:
        c, s = parse_stem(p.stem)
        if c is None or s is None:
            continue
        index.setdefault((norm(c), norm(s)), []).append(p)

    out_dir = Path(args.out)
    if not args.dry_run and not args.in_place:
        out_dir.mkdir(parents=True, exist_ok=True)

    done = missing = failed = 0
    for cluster, segment, desc in rows:
        key = (norm(cluster), norm(segment))
        matches = index.get(key, [])
        if not matches:
            print(f"  - no file for cluster={cluster} segment={segment}")
            missing += 1
            continue
        if len(matches) > 1:
            print(f"  ! cluster={cluster} segment={segment}: {len(matches)} matches, processing all")

        for src in matches:
            dst_name = new_stem(src.stem, desc) + src.suffix.lower()
            dst = (src.parent if args.in_place else out_dir) / dst_name
            print(f"  {src.name}  ->  {dst_name}   ({args.gain:+g} dB)")
            if args.dry_run:
                done += 1
                continue
            try:
                if args.in_place:
                    tmp = src.with_name(src.name + ".tmp")
                    clipped = write_with_gain(src, tmp, args.gain)
                    shutil.move(str(tmp), str(dst))
                    if dst != src and src.exists():
                        src.unlink()
                else:
                    clipped = write_with_gain(src, dst, args.gain)
                if clipped:
                    print("      (clipping - lower the gain)")
                done += 1
            except Exception as e:
                print(f"      FAILED: {e}")
                failed += 1

    print(f"\n{done} written, {missing} not found, {failed} failed"
          + ("  [dry run]" if args.dry_run else ""))


if __name__ == "__main__":
    main()
