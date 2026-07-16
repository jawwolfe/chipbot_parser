import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf
# Clustering and reduction
import umap
from sklearn.cluster import HDBSCAN
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

def extract_embeddings_and_detect(file_path, analyzer, min_conf=0.40):
    """
    Runs species detection using the customized species list Analyzer,
    then dynamically builds a patched Analyzer to extract 1024-D embeddings.
    """
    # 1. Standard detection using your custom-list analyzer
    detection_recording = Recording(analyzer=analyzer, path=str(file_path), min_conf=min_conf)
    detection_recording.analyze()
    detections = detection_recording.detections

    # 2. Safely swap the interpreter ONLY while creating and running the embedding analyzer
    original_interpreter = tf.lite.Interpreter

    class EmbeddingSafeInterpreter(original_interpreter):
        def __init__(self, *args, **kwargs):
            kwargs['experimental_preserve_all_tensors'] = True
            super().__init__(*args, **kwargs)

    # Apply the patch right before instantiating the embedding engine
    tf.lite.Interpreter = EmbeddingSafeInterpreter

    try:
        # Moving the instantiation HERE forces the embedding engine to preserve intermediate layers
        embedding_analyzer = Analyzer()
        embedding_recording = Recording(analyzer=embedding_analyzer, path=str(file_path))
        embedding_recording.analyze()
        embedding_recording.extract_embeddings()

        raw_embeddings = embedding_recording.embeddings  # List of raw tensors/dicts in TF 2.16+
        chunks = embedding_recording.chunks
    finally:
        # IMMEDIATELY restore the original interpreter to keep the environment clean
        tf.lite.Interpreter = original_interpreter

        # --- Clean the extracted dictionary arrays (TF 2.16+ wrapper fix) ---
        cleaned_embeddings = []
        for emb in raw_embeddings:
            if emb is None:
                cleaned_embeddings.append(None)
                continue

            try:
                # 1. If it's a TensorFlow EagerTensor (has .numpy() method)
                if hasattr(emb, 'numpy'):
                    cleaned_embeddings.append(emb.numpy().flatten())

                # 2. If it's a dictionary (older BirdNET-Analyzer formats)
                elif isinstance(emb, dict):
                    # Try getting 'array', fallback to 'embeddings', or the first value
                    val = emb.get('array') or emb.get('embeddings') or list(emb.values())[0]
                    cleaned_embeddings.append(np.asarray(val).flatten())

                # 3. If it's already a numpy array
                elif isinstance(emb, np.ndarray):
                    cleaned_embeddings.append(emb.flatten())

                # 4. Fallback: Try converting lists, tuples, or any other iterable directly
                else:
                    arr = np.asarray(emb)
                    if arr.size > 0:
                        cleaned_embeddings.append(arr.flatten())
                    else:
                        cleaned_embeddings.append(None)

            except Exception as e:
                # If coercion fails for a specific chunk, log it and keep going
                print(f"   [Warning] Failed to parse embedding element: {e}")
                cleaned_embeddings.append(None)

    chunks_metadata = []

    # Map chunks to detections
    for i, chunk in enumerate(chunks):
        start_time = i * 3.0
        end_time = start_time + 3.0

        # Pull the matching 1024-D vector
        feat_vector = cleaned_embeddings[i] if i < len(cleaned_embeddings) else None
        if feat_vector is None or feat_vector.shape[0] != 1024:
            continue

        # Look for custom-filtered detections in this 3-second window
        chunk_detections = [
            d for d in detections
            if abs(d['start_time'] - start_time) < 1.5
        ]

        label = "Unidentified/Ambient"
        confidence = 0.0
        if chunk_detections:
            best_det = max(chunk_detections, key=lambda x: x['confidence'])
            label = f"{best_det['common_name']} ({best_det['scientific_name']})"
            confidence = best_det['confidence']

        chunks_metadata.append({
            "file": file_path.name,
            "start_time": start_time,
            "end_time": end_time,
            "birdnet_label": label,
            "confidence": confidence
        })

    # Filter out any None values from the final array to prevent stacking errors downstream
    valid_embeddings = [e for e in cleaned_embeddings if e is not None and e.shape[0] == 1024]

    return detections, np.array(valid_embeddings), chunks_metadata


def run_pipeline(audio_dir, output_dir, species_list_path=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus}\n")
    else:
        print("No GPU detected. Defaulting to CPU.\n")

    # Initialize the customized detector
    print("Initializing customized species list analyzer...")
    analyzer = Analyzer(custom_species_list_path=species_list_path)

    audio_extensions = {".wav"}
    audio_dir_path = Path(audio_dir)

    audio_files = [f for f in audio_dir_path.iterdir() if f.suffix.lower() in audio_extensions]
    if not audio_files:
        print(f"No matching audio files found.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_txt_path = output_path / f"detection_results_{timestamp}.txt"

    all_embeddings = []
    all_metadata = []

    with open(results_txt_path, "w", encoding="utf-8") as f_out:
        for index, file_path in enumerate(audio_files, 1):
            print(f"[{index}/{len(audio_files)}] Processing: {file_path.name}...")
            f_out.write(f"=== File: {file_path.name} ===\n")

            try:
                # Call the updated extraction function (removed the extra analyzer parameter)
                detections, embeddings, metadata = extract_embeddings_and_detect(
                    file_path, analyzer, min_conf=0.4
                )

                if len(embeddings) > 0:
                    all_embeddings.append(embeddings)
                    all_metadata.extend(metadata)

                # Write text detections
                if not detections:
                    f_out.write("No detections found.\n")
                else:
                    for detection in detections:
                        result_line = (
                            f"Time: {detection['start_time']:.1f}s - {detection['end_time']:.1f}s | "
                            f"Species: {detection['common_name']} ({detection['scientific_name']}) | "
                            f"Confidence: {detection['confidence']:.2%}\n"
                        )
                        f_out.write(result_line)

            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {e}\n"
                f_out.write(error_msg)
                print(error_msg)

            f_out.write("\n" + "=" * 50 + "\n\n")

        if all_embeddings:
            print("\n--- Running Dimensionality Reduction & Clustering ---")
            X = np.vstack(all_embeddings)

            # 1. Normalize embeddings (essential for cosine distance)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_normalized = np.where(norms == 0, X, X / norms)

            # 2. Reduce to a moderate-dimensional space FIRST, then cluster on that.
            # Clustering directly on 1024-D embeddings collapses HDBSCAN to ~1 cluster + noise.
            print("Reducing embeddings to 10-D with UMAP for clustering...")
            cluster_reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.0,
                n_components=8,
                metric='cosine',
                random_state=42
            )
            X_umap = cluster_reducer.fit_transform(X_normalized)

            print("Clustering reduced embeddings with HDBSCAN...")
            clusterer = HDBSCAN(
                min_cluster_size=4,
                min_samples=3,
                metric='euclidean'  # fixed typo; valid since UMAP output is euclidean space
            )
            cluster_labels = clusterer.fit_predict(X_umap)

            # 3. Separate UMAP run, purely for 2D visualization
            print("Projecting embeddings to 2D with UMAP for visualization...")
            viz_reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.0,
                n_components=2,
                metric='cosine',
                random_state=42
            )
            X_2d = viz_reducer.fit_transform(X_normalized)

            # 4. Build DataFrame
            df = pd.DataFrame(all_metadata)
            df['umap_x'] = X_2d[:, 0]
            df['umap_y'] = X_2d[:, 1]
            df['cluster'] = cluster_labels

        csv_path = output_path / f"acoustic_clusters_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Clustering complete! Detailed data saved to: {csv_path}")

        unidentified_clusters = df[(df['birdnet_label'] == "Unidentified/Ambient") & (df['cluster'] != -1)]
        if not unidentified_clusters.empty:
            print(f"\n[AHA!] Found {len(unidentified_clusters)} unidentified segments that clustered together!")
            print(unidentified_clusters[['file', 'start_time', 'cluster']].head(10).to_string(index=False))
        else:
            print("\nNo distinct clusters of unidentified audio found.")


if __name__ == "__main__":
    AUDIO_DIRECTORY = "data/files"
    OUTPUT_DIRECTORY = "data/results"
    SPECIES_LIST = "data/species/philippines.txt"

    run_pipeline(audio_dir=AUDIO_DIRECTORY, output_dir=OUTPUT_DIRECTORY,
                 species_list_path=SPECIES_LIST)