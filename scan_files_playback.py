import os
import re
import time
from collections import defaultdict
from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path


def parse_filename(filename):
    """
    Parses filenames structured like: Name_Species_Num1_Num2_Gain.wav
    Example: Watercock_(Gallicrex_cinerea)_178_34_-24.wav
    Returns (name, num2) if valid, otherwise (None, None).
    """
    pattern = r"^([^_]+)_(.+)_(\d+)_(\d+)_(-?\d+)\.wav$"
    match = re.match(pattern, filename, re.IGNORECASE)

    if match:
        name = match.group(1)
        num2 = int(match.group(4))  # Number before the last underscore
        return name, num2
    return None, None


def play_top_3_num2_per_name(directory_path, gain_db=15.0, top_k=3, max_duration_ms=4000, pause_sec=5):
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # Dictionary to group files: { name: [(num2, file_path, sound), ...] }
    grouped_files = defaultdict(list)

    print("Scanning directory and checking audio durations...")

    # 1. Scan directory, filter by duration, and group files by distinct Name
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.wav'):
                name, num2 = parse_filename(file)
                if name is not None:
                    file_path = os.path.join(root, file)

                    try:
                        # Load sound metadata to check duration
                        sound = AudioSegment.from_wav(file_path)
                        duration_ms = len(sound)

                        # Exclude files longer than max_duration_ms (4,000 ms)
                        if duration_ms > max_duration_ms:
                            print(f"  [Skipped - Too Long ({duration_ms}ms)]: {file}")
                            continue

                        grouped_files[name].append((num2, file_path, sound))

                    except Exception as e:
                        print(f"  Could not read file {file}: {e}")

    if not grouped_files:
        print("No matching .wav files under the duration threshold were found.")
        return

    # 2. Process each distinct Name
    for name, entries in grouped_files.items():
        # Sort by num2 descending to get highest values first
        entries.sort(key=lambda x: x[0], reverse=True)

        # Take up to top_k entries (e.g., top 3)
        top_entries = entries[:top_k]

        print(f"\n==========================================")
        print(f"Name: {name} (Playing Top {len(top_entries)} files)")
        print(f"==========================================")

        for rank, (num2, file_path, sound) in enumerate(top_entries, start=1):
            file_name = os.path.basename(file_path)
            duration_sec = len(sound) / 1000.0
            print(f" #{rank} [Num2: {num2} | Duration: {duration_sec:.2f}s] -> {file_name}")

            try:
                louder_sound = sound + gain_db
                play(louder_sound)

                # 5-second pause after playback finishes
                print(f"    Pausing for {pause_sec} seconds...")
                time.sleep(pause_sec)

            except Exception as e:
                print(f"    Failed to play {file_name}: {e}")

if __name__ == "__main__":
    # Replace with your target folder path
    AUDIO_DIR = Path("/mnt/c/") / Path("temp/CHIPBOT_DATA_ROOT") / Path("links_detections/Philippines_Cebu_Pacijan-Lake-Danao-Marsh-Cental_2026-09-11-160015_2026-09-11-201518")


    # Set high gain boost (e.g., +12 dB to +20 dB)
    # Caution: Values above +15 dB may cause clipping/distortion depending on the source audio.
    GAIN_BOOST = 25.0
    MAX_DURATION_MS = 6000  # 4,000 ms = 4 seconds
    PAUSE_SEC = 3  # Pause duration in seconds

    play_top_3_num2_per_name(
        AUDIO_DIR,
        gain_db=GAIN_BOOST,
        top_k=3,
        max_duration_ms=MAX_DURATION_MS,
        pause_sec=PAUSE_SEC
    )