import os
from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path


def play_wavs_with_gain(directory_path, gain_db=15.0):
    """
    Loops through a directory and plays all .wav files with applied gain.

    :param directory_path: Path to the directory containing .wav files.
    :param gain_db: Gain boost in decibels (dB). Default is +12 dB.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # Recursively find all .wav files
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                print(f"Playing: {file_path} (+{gain_db} dB gain)")

                try:
                    # Load audio file
                    sound = AudioSegment.from_wav(file_path)

                    # Apply high gain (positive values increase volume in dB)
                    louder_sound = sound + gain_db

                    # Play the modified audio
                    play(louder_sound)

                except Exception as e:
                    print(f"Failed to play {file}: {e}")


if __name__ == "__main__":
    # Replace with your target folder path
    AUDIO_DIR = Path("/mnt/c/") / Path("temp/CHIPBOT_DATA_ROOT") / Path("links_detections/Philippines_Cebu_Pacijan-Lake-Danao-Marsh-Cental_2026-09-11-160015_2026-09-11-201518")


    # Set high gain boost (e.g., +12 dB to +20 dB)
    # Caution: Values above +15 dB may cause clipping/distortion depending on the source audio.
    GAIN_BOOST = 20.0

    play_wavs_with_gain(AUDIO_DIR, gain_db=GAIN_BOOST)