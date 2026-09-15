import os
import re
import sys
import time
from collections import defaultdict
from pydub import AudioSegment
import pygame
from pathlib import Path

def parse_filename(filename):
    """
    Parses filenames structured like: Name_(Species)_Num1_Num2_Gain.wav
    Example: Clamorous_Reed_Warbler_(Acrocephalus_stentoreus)_46_40_-36.wav
    """
    pattern = r"^(.+?)_\((.+?)\)_(\d+)_(\d+)_(-?\d+)\.wav$"
    match = re.match(pattern, filename, re.IGNORECASE)

    if match:
        raw_name = match.group(1)
        name = raw_name.replace("_", " ").strip()
        num2 = int(match.group(4))
        return name, num2
    return None, None


def render_status_window(screen, font, species_name, track_info, status_text):
    """Updates the Pygame window with active playback status."""
    screen.fill((30, 30, 30))  # Dark gray background

    txt_species = font.render(f"Species: {species_name}", True, (255, 255, 255))
    txt_track = font.render(track_info, True, (200, 200, 200))
    txt_status = font.render(status_text, True, (0, 255, 150))
    txt_controls = font.render("[SPACE] Pause/Resume  |  [S] Skip", True, (150, 150, 150))

    screen.blit(txt_species, (20, 20))
    screen.blit(txt_track, (20, 50))
    screen.blit(txt_status, (20, 90))
    screen.blit(txt_controls, (20, 130))

    pygame.display.flip()


def play_audio_with_controls(screen, font, file_path, gain_db, pause_between_tracks_sec, species_name, track_info):
    """
    Plays audio and uses Pygame event polling for responsive Spacebar / S detection.
    """
    temp_file = "temp_boosted.wav"

    try:
        # Generate temporary audio with gain boost
        sound = AudioSegment.from_wav(file_path) + gain_db
        sound.export(temp_file, format="wav")

        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        is_paused = False
        status_msg = "Playing..."
        render_status_window(screen, font, species_name, track_info, status_msg)

        # Main playback event loop
        while pygame.mixer.music.get_busy() or is_paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if is_paused:
                            pygame.mixer.music.unpause()
                            is_paused = False
                            status_msg = "Playing..."
                        else:
                            pygame.mixer.music.pause()
                            is_paused = True
                            status_msg = "PAUSED"
                        render_status_window(screen, font, species_name, track_info, status_msg)

                    elif event.key == pygame.K_s:
                        status_msg = "Skipped track"
                        render_status_window(screen, font, species_name, track_info, status_msg)
                        pygame.mixer.music.stop()
                        break

            pygame.time.Clock().tick(30)  # Keep loop responsive (30 FPS)

    except Exception as e:
        print(f"    Error playing file {file_path}: {e}")

    finally:
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except PermissionError:
                pass

    # Inter-track delay loop with full event checking
    if pause_between_tracks_sec > 0:
        start_time = time.time()
        is_paused = False

        while (time.time() - start_time) < pause_between_tracks_sec:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        is_paused = not is_paused
                        if is_paused:
                            status_msg = "PAUSED (Inter-track Delay)"
                        else:
                            status_msg = "Waiting for next track..."
                        render_status_window(screen, font, species_name, track_info, status_msg)

                    elif event.key == pygame.K_s:
                        status_msg = "Skipped delay"
                        render_status_window(screen, font, species_name, track_info, status_msg)
                        return

            if is_paused:
                # Freeze timer during pause
                start_time += 0.05
            else:
                remaining = max(0, int(pause_between_tracks_sec - (time.time() - start_time)))
                status_msg = f"Waiting {remaining}s for next track..."
                render_status_window(screen, font, species_name, track_info, status_msg)

            time.sleep(0.05)


def play_top_3_num2_per_name(
        directory_path,
        gain_db=15.0,
        top_k=3,
        max_duration_ms=4000,
        pause_sec=5,
        start_from_name=None
):
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # Initialize Pygame Display and Audio
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((500, 180))
    pygame.display.set_caption("Audio Player Controls")
    font = pygame.font.SysFont(None, 24)

    grouped_files = defaultdict(list)
    print("Scanning directory and checking audio durations...")

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.wav'):
                name, num2 = parse_filename(file)
                if name is not None:
                    file_path = os.path.join(root, file)

                    try:
                        sound = AudioSegment.from_wav(file_path)
                        duration_ms = len(sound)

                        if duration_ms > max_duration_ms:
                            print(f"  [Skipped - Exceeds {max_duration_ms / 1000}s]: {file}")
                            continue

                        grouped_files[name].append((num2, file_path, duration_ms))

                    except Exception as e:
                        print(f"  Could not read file {file}: {e}")

    if not grouped_files:
        print("No matching .wav files found under the criteria.")
        pygame.quit()
        return

    sorted_names = sorted(grouped_files.keys())

    normalized_start_name = None
    if start_from_name:
        normalized_start_name = start_from_name.replace("_", " ").strip().lower()

    should_play = normalized_start_name is None

    for name in sorted_names:
        if not should_play:
            if name.lower() == normalized_start_name:
                should_play = True
            else:
                print(f"Skipping '{name}' (waiting for target start species)")
                continue

        entries = grouped_files[name]
        entries.sort(key=lambda x: x[0], reverse=True)
        top_entries = entries[:top_k]

        print(f"\n==========================================")
        print(f"Species: {name} (Playing Top {len(top_entries)} files)")
        print(f"==========================================")

        for rank, (num2, file_path, duration_ms) in enumerate(top_entries, start=1):
            file_name = os.path.basename(file_path)
            duration_sec = duration_ms / 1000.0
            track_info = f"#{rank} | Num2: {num2} | {file_name}"

            print(f" #{rank} [Num2: {num2} | Duration: {duration_sec:.2f}s] -> {file_name}")

            play_audio_with_controls(
                screen,
                font,
                file_path,
                gain_db,
                pause_sec,
                name,
                track_info
            )

    pygame.quit()

if __name__ == "__main__":
    # Replace with your target folder path
    AUDIO_DIR = Path("/mnt/c/") / Path("temp/CHIPBOT_DATA_ROOT") / Path("links_detections/Philippines_Cebu_Pacijan-Lake-Danao-Marsh-Cental_2026-09-11-160015_2026-09-11-201518")
    GAIN_BOOST = 15.0  # Gain boost in dB
    MAX_DURATION_MS = 45000  # 4,000 ms = 4 seconds
    PAUSE_SEC = 3  # Pause duration in seconds

    # Set to a species name to start playback from that point.
    # Set to None to play from the beginning.
    START_SPECIES = "Clamorous_Reed_Warbler"

    play_top_3_num2_per_name(
        AUDIO_DIR,
        gain_db=GAIN_BOOST,
        top_k=3,
        max_duration_ms=MAX_DURATION_MS,
        pause_sec=PAUSE_SEC,
        start_from_name=START_SPECIES
    )