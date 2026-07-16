import os
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pathlib import Path



def run_pipeline(audio_dir, output_dir, species_list_path=None):
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected and will be used: {gpus}")
    else:
        print("No GPU detected. Defaulting to CPU. Check your CUDA/cuDNN installation.")

    analyzer = Analyzer(
        custom_species_list_path=species_list_path
    )

    recording = Recording(
        analyzer=analyzer,
        path=audio_dir,
        min_conf=0.25  # Custom minimum confidence threshold
    )
    recording.analyze()

    for detection in recording.detections:
        print(
            f"Time: {detection['start_time']:.1f}s - {detection['end_time']:.1f}s | "
            f"Species: {detection['common_name']} ({detection['scientific_name']}) | "
            f"Confidence: {detection['confidence']:.2%}"
        )


if __name__ == "__main__":
    AUDIO_DIRECTORY = "data/file/aw_chipbot_01_2026-07-13_06_11_02_39.875651_-86.284142.wav"
    OUTPUT_DIRECTORY = "data/results"
    SPECIES_LIST = "data/species/others.txt"  # Path to your manual species list

    run_pipeline(audio_dir=AUDIO_DIRECTORY, output_dir=OUTPUT_DIRECTORY,
                 species_list_path=SPECIES_LIST)