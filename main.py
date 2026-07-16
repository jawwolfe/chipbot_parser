from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pathlib import Path
from datetime import datetime


def run_pipeline(audio_dir, output_dir, species_list_path=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # GPU check
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected and will be used: {gpus}\n")
    else:
        print("No GPU detected. Defaulting to CPU. Check your CUDA/cuDNN installation.\n")

    # Initialize analyzer once to save memory and time
    analyzer = Analyzer(
        custom_species_list_path=species_list_path
    )

    # Define supported audio extensions
    audio_extensions = {".wav"}

    # Find all audio files in the directory
    audio_dir_path = Path(audio_dir)
    if not audio_dir_path.is_dir():
        raise ValueError(f"The provided path '{audio_dir}' is not a directory.")

    audio_files = [f for f in audio_dir_path.iterdir() if f.suffix.lower() in audio_extensions]

    if not audio_files:
        print(f"No matching audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files to process.")

    # Path for the consolidated output text file
    results_txt_path = output_path / "detection_results.txt"

    # Open the text file to write results
    # Generate a unique timestamp (e.g., "20260715_161002")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_txt_path = output_path / f"detection_results_{timestamp}.txt"
    with open(results_txt_path, "w", encoding="utf-8") as f_out:
        for index, file_path in enumerate(audio_files, 1):
            print(f"[{index}/{len(audio_files)}] Processing: {file_path.name}...")
            f_out.write(f"=== File: {file_path.name} ===\n")

            try:
                # Run BirdNET analysis on the individual file
                recording = Recording(
                    analyzer=analyzer,
                    path=str(file_path),
                    min_conf=0.10  # Custom minimum confidence threshold
                )
                recording.analyze()

                # Write results
                if not recording.detections:
                    f_out.write("No detections found.\n")
                else:
                    for detection in recording.detections:
                        result_line = (
                            f"Time: {detection['start_time']:.1f}s - {detection['end_time']:.1f}s | "
                            f"Species: {detection['common_name']} ({detection['scientific_name']}) | "
                            f"Confidence: {detection['confidence']:.2%}\n"
                        )
                        # Write to file
                        f_out.write(result_line)
                        # Also print to console for live monitoring
                        print(f"  -> {result_line.strip()}")

            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {e}\n"
                f_out.write(error_msg)
                print(error_msg)

            f_out.write("\n" + "=" * 50 + "\n\n")

    print(f"\nProcessing complete! Results saved to: {results_txt_path}")


if __name__ == "__main__":
    # Point this to the DIRECTORY containing your audio files, not a single file
    AUDIO_DIRECTORY = "data/files"
    OUTPUT_DIRECTORY = "data/results"
    SPECIES_LIST = "data/species/midwest.txt"  # Path to your manual species list

    run_pipeline(audio_dir=AUDIO_DIRECTORY, output_dir=OUTPUT_DIRECTORY,
                 species_list_path=SPECIES_LIST)