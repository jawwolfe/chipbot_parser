import os
import re
import sys
import time
from pathlib import Path
from pydub import AudioSegment
import pygame


def parse_num2(filename):
    """
    Parses filenames to extract Num2 (the number right before the last underscore).
    Example: Watercock_(Gallicrex_cinerea)_178_34_-24.wav -> returns 34
    Returns integer Num2 if valid, otherwise None.
    """
    pattern = r"^.+?_(\d+)_(-?\d+)\.wav$"
    match = re.match(pattern, filename, re.IGNORECASE)

    if match:
        return int(match.group(1))
    return None


def render_status_window(screen, font, dir_name, track_info, status_text):
    """Updates the Pygame window with active playback status."""
    screen.fill((30, 30, 30))  # Dark gray background

    txt_dir = font.render(f"Directory: {dir_name}", True, (255, 255, 255))
    txt_track = font.render(track_info, True, (200, 200, 200))
    txt_status = font.render(status_text, True, (0, 255, 150))
    txt_controls = font.render("[SPACE] Pause/Resume  |  [S] Skip", True, (150, 150, 150))

    screen.blit(txt_dir, (20, 20))
    screen.blit(txt_track, (20, 50))
    screen.blit(txt_status, (20, 90))
    screen.blit(txt_controls, (20, 130))

    pygame.display.flip()


def play_audio_with_controls(screen, font, file_path, gain_db, pause_between_tracks_sec, dir_name, track_info):
    """
    Plays audio and uses Pygame event polling for responsive Spacebar / S detection.
    """
    temp_file = "temp_boosted.wav"

    try:
        sound = AudioSegment.from_wav(file_path) + gain_db
        sound.export(temp_file, format="wav")

        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        is_paused = False
        status_msg = "Playing..."
        render_status_window(screen, font, dir_name, track_info, status_msg)

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
                        render_status_window(screen, font, dir_name, track_info, status_msg)

                    elif event.key == pygame.K_s:
                        status_msg = "Skipped track"
                        render_status_window(screen, font, dir_name, track_info, status_msg)
                        pygame.mixer.music.stop()
                        break

            pygame.time.Clock().tick(30)

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

    # Inter-track delay loop
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
                        render_status_window(screen, font, dir_name, track_info, status_msg)

                    elif event.key == pygame.K_s:
                        status_msg = "Skipped delay"
                        render_status_window(screen, font, dir_name, track_info, status_msg)
                        return

            if is_paused:
                start_time += 0.05
            else:
                remaining = max(0, int(pause_between_tracks_sec - (time.time() - start_time)))
                status_msg = f"Waiting {remaining}s for next track..."
                render_status_window(screen, font, dir_name, track_info, status_msg)

            time.sleep(0.05)

def natural_sort_key(s):
    """
    Splits string into numbers and text so directories sort naturally:
    '1', '2', '3', '10', '101' instead of '1', '10', '101', '2', '3'.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def play_top_3_per_directory(
        directory_path,
        gain_db=15.0,
        top_k=3,
        max_duration_ms=45000,
        pause_sec=3,
        start_from_dir=None
):
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((500, 180))
    pygame.display.set_caption("Directory Audio Player Controls")
    font = pygame.font.SysFont(None, 24)

    normalized_start_dir = None
    if start_from_dir and start_from_dir.strip():
        normalized_start_dir = start_from_dir.strip().lower()

    should_play = normalized_start_dir is None

    # Walk directory tree folder by folder
    for root, dirs, files in os.walk(directory_path):
        # Sort subdirectories naturally (1, 2, 3 ... 10, 11 ... 100)
        dirs.sort(key=natural_sort_key)

        folder_name = os.path.basename(root) or os.path.basename(directory_path)

        # Check start directory condition
        if not should_play:
            if folder_name.lower() == normalized_start_dir or root.lower().endswith(normalized_start_dir):
                should_play = True
            else:
                print(f"Skipping directory '{folder_name}' (waiting for start target)")
                continue

        dir_files = []
        for file in files:
            if file.lower().endswith('.wav'):
                num2 = parse_num2(file)
                if num2 is not None:
                    file_path = os.path.join(root, file)
                    try:
                        sound = AudioSegment.from_wav(file_path)
                        duration_ms = len(sound)

                        if duration_ms > max_duration_ms:
                            print(f"  [Skipped - Exceeds {max_duration_ms / 1000}s]: {file}")
                            continue

                        dir_files.append((num2, file_path, duration_ms))
                    except Exception as e:
                        print(f"  Could not read file {file}: {e}")

        # If valid files exist in this folder, sort and play top K
        if dir_files:
            dir_files.sort(key=lambda x: x[0], reverse=True)
            top_entries = dir_files[:top_k]

            print(f"\n==========================================")
            print(f"Directory: {root} (Playing Top {len(top_entries)} files)")
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
                    folder_name,
                    track_info
                )

    pygame.quit()


if __name__ == "__main__":
    AUDIO_DIR = Path("/mnt/c/") / Path("temp/CHIPBOT_DATA_ROOT") / Path("links_clusters/run_2")
    GAIN_BOOST = 25.0
    MAX_DURATION_MS = 250000  # 45 seconds max duration
    PAUSE_SEC = 1  # Pause duration in seconds

    # Set to a directory folder name to start playback, or "" / None to play from beginning
    START_DIR = "cluster_240_4min_4seg"

    play_top_3_per_directory(
        AUDIO_DIR,
        gain_db=GAIN_BOOST,
        top_k=3,
        max_duration_ms=MAX_DURATION_MS,
        pause_sec=PAUSE_SEC,
        start_from_dir=START_DIR
    )