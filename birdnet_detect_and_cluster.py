import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import natsort
import tensorflow as tf
# Clustering and reduction
import umap
from sklearn.cluster import HDBSCAN
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import shutil, os, csv, sys, wave, re
from collections import defaultdict

FILE_PREFIX = "cluster_"
MAX_SPECIES_NAME_LENGTH = 120
IGNORED_LABEL = "unidentified/ambient"

class BirdNetParserBase:
    def __init__(self, logger):
        self.logger = logger

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

class BirdNetParser(BirdNetParserBase):
    def __init__(self, logger, external_drive, audio_path, output_path, min_confidence_input, min_confidence_output,
                 species_list, gap_ms, hdbscan_clusters, umap, umap_second, analysis_run_text, analyze_file_group):
        self.external_drive = external_drive
        self.audio_path = audio_path
        self.output_path = output_path
        self.min_confidence_input = min_confidence_input
        self.min_confidence_output = min_confidence_output
        self.species_list_path = species_list
        self.gap_ms = gap_ms
        self.umap = umap
        self.umap_second = umap_second
        self.hdbscan_clusters = hdbscan_clusters
        self.analysis_run_text = analysis_run_text
        self.analyze_file_group = analyze_file_group
        BirdNetParserBase.__init__(self, logger=logger)

    def read_rows(self, csv_path):
        rows = []
        skipped_blank = 0
        skipped_low_confidence = 0

        with open(csv_path, newline="") as f:
            # Custom dict reader wrapper to handle case-insensitive headers just in case
            raw_reader = csv.DictReader(f)
            if not raw_reader.fieldnames:
                sys.exit("CSV appears to be empty or missing a header row.")

            # Map headers to lowercase to prevent "birdnet_label" vs "BirdNet_label" mismatches
            header_map = {name.lower().strip(): name for name in raw_reader.fieldnames}

            required = {"file", "start_time", "end_time", "cluster", "birdnet_label"}
            missing = required - set(header_map.keys())
            if missing:
                sys.exit(f"CSV is missing required columns: {sorted(missing)}")

            for i, raw_row in enumerate(raw_reader, start=2):  # header is line 1
                # Reconstruct row using lowercase keys for safe access
                row = {k.lower().strip(): v for k, v in raw_row.items() if k}

                label = (row.get("birdnet_label") or "").strip()
                if not label:
                    skipped_blank += 1
                    continue

                try:
                    start = float(row["start_time"])
                    end = float(row["end_time"])
                    cluster = row["cluster"].strip()
                    # Safely get confidence if it exists
                    conf_val = row.get("confidence")
                    conf = float(conf_val) if conf_val not in (None, "") else None
                except ValueError as e:
                    sys.exit(f"CSV row {i}: could not parse numeric field ({e})")

                if end <= start:
                    print(f"Warning: row {i} has end_time <= start_time, skipping", file=sys.stderr)
                    continue

                # Determine if this row belongs to the ambient/unidentified category
                is_ambient = label.lower() == IGNORED_LABEL

                # CRITICAL FIX: Only apply min_confidence to IDENTIFIED species.
                # Ambient/Unidentified clips often have 0.0 or low confidence scores.
                if not is_ambient and self.min_confidence_output is not None and conf is not None and conf < self.min_confidence_output:
                    skipped_low_confidence += 1
                    continue

                rows.append({
                    "file": row["file"].strip(),
                    "start": start,
                    "end": end,
                    "cluster": cluster,
                    "confidence": conf,
                    "label": label,  # Keeps original casing for filename
                    "is_ambient": is_ambient
                })

        print(f"Loaded {len(rows)} total valid rows.")
        print(f"  - Skipped {skipped_blank} blank label rows.")
        print(f"  - Skipped {skipped_low_confidence} identified species rows below {self.min_confidence_output} confidence.")
        return rows

    def sanitize_for_filename(self, label):
        """Turn a birdnet_label into a filesystem-safe chunk for use in a filename."""
        cleaned = label.strip()
        if "_" in cleaned and cleaned.count("_") == 1:
            _, _, common = cleaned.partition("_")
            if common:
                cleaned = common
        cleaned = cleaned.replace("/", "-").replace(" ", "_")
        cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
        return cleaned or "unknown_species"

    def all_species_slug(self, group, max_length=120):
        """Return a filename-safe chunk listing distinct species in a cluster."""
        slugs = sorted({self.sanitize_for_filename(r["label"]) for r in group})
        full = "+".join(slugs)
        if len(full) <= max_length:
            return full

        kept = []
        length = 0
        for slug in slugs:
            added = (1 if kept else 0) + len(slug)
            if length + added > max_length:
                break
            kept.append(slug)
            length += added
        remaining = len(slugs) - len(kept)
        if not kept:
            return slugs[0][:max_length]
        return "+".join(kept) + f"+{remaining}_more"

    def extract_segment_bytes(self, wf, params, start_time, end_time):
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

    def write_cluster_wavs(self, by_cluster, cache, destination_dir, file_prefix, max_species_name_length,
                           audio_format_params,
                           gap_bytes):
        """Helper to process a dictionary of clusters and write out combined WAV files."""
        nchannels, sampwidth, framerate = audio_format_params

        for cluster, group in sorted(by_cluster.items(), key=lambda kv: kv[0]):
            labels = sorted({r["label"] for r in group})
            species_slug = self.all_species_slug(group, max_length=max_species_name_length)
            out_path = os.path.join(destination_dir, f"{file_prefix}{cluster}_{species_slug}.wav")
            total_duration = sum(r["end"] - r["start"] for r in group)

            print(f"  cluster {cluster}: {len(group)} segment(s), "
                  f"~{total_duration:.1f}s, labels: {', '.join(labels)} -> {out_path}")

            if len(os.path.abspath(out_path)) > 245:
                print(f"    Warning: full output path is {len(os.path.abspath(out_path))} chars long; "
                      f"this may fail on Windows (260-char limit).", file=sys.stderr)

            with wave.open(out_path, "wb") as out_wf:
                out_wf.setnchannels(nchannels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(framerate)

                for idx, row in enumerate(group):
                    wf = cache.get(row["file"])
                    params = cache.params(row["file"])
                    data = self.extract_segment_bytes(wf, params, row["start"], row["end"])
                    out_wf.writeframes(data)
                    if gap_bytes and idx != len(group) - 1:
                        out_wf.writeframes(gap_bytes)


    def extract_embeddings_and_detect(self, file_path, analyzer):
        """
        Runs species detection using the customized species list Analyzer,
        then dynamically builds a patched Analyzer to extract 1024-D embeddings.
        """
        # 1. Standard detection using your custom-list analyzer
        detection_recording = Recording(analyzer=analyzer, path=str(file_path), min_conf=self.min_confidence_input)
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


    def run_pipeline(self):
        output_dir = Path(self.output_path)
        external_dir = Path(self.external_drive + '://')

        # GPU check
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU detected: {gpus}\n")
        else:
            print("No GPU detected. Defaulting to CPU.\n")

        # Initialize the customized detector
        print("Initializing customized species list analyzer...")
        analyzer = Analyzer(custom_species_list_path=self.species_list_path)

        if self.analyze_file_group:
            # doing a reanalysis now of existing files
            # source path is in audio dir processed then name of file group
            source_audio_dir = Path(self.audio_path + '\\processed\\' + self.analyze_file_group)
            run_suffix = self.analysis_run_text
        else:
            # running new file group fom external drive
            valid_extensions = {".wav", ".txt"}
            for item in external_dir.iterdir():
                # Only process non-recursive files matching valid extensions
                if item.is_file() and item.suffix.lower() in valid_extensions:
                    if item.stat().st_size > 0:
                        shutil.move(item, Path(self.audio_path) / item.name)
                    else:
                        item.unlink()
            source_audio_dir = Path(self.audio_path)
            run_suffix = 'initial'

        audio_extensions = {".wav"}
        audio_dir_path = Path(source_audio_dir)
        audio_files = [f for f in audio_dir_path.iterdir() if f.suffix.lower() in audio_extensions]
        if not audio_files:
            print(f"No matching audio files found.")
            return

        audio_files = natsort.natsorted(audio_files, key=lambda x: str(x))
        first_file_name = audio_files[0].stem
        analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_results = Path(Path(self.output_path) / f"{first_file_name}_{analysis_timestamp}_{run_suffix}")
        output_path_results.mkdir(parents=True, exist_ok=True)
        detection_file_path = output_path_results / f"detection_results_{first_file_name}_{analysis_timestamp}_{run_suffix}.txt"

        all_embeddings = []
        all_metadata = []
        with open(detection_file_path, "w", encoding="utf-8") as f_out:
            for index, file_path in enumerate(audio_files, 1):
                print(f"[{index}/{len(audio_files)}] Processing: {file_path.name}...")
                f_out.write(f"=== File: {file_path.name} ===\n")

                try:
                    detections, embeddings, metadata = self.extract_embeddings_and_detect(
                        file_path, analyzer)

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
                X_all = np.vstack(all_embeddings)
                df_all = pd.DataFrame(all_metadata)

                # Only cluster chunks BirdNET did NOT confidently identify.
                identified_mask = (df_all['birdnet_label'] == "Unidentified/Ambient").to_numpy() == False
                X_identified = X_all[identified_mask]
                X_unidentified = X_all[~identified_mask]
                df_identified = df_all[identified_mask].copy()
                df_unidentified = df_all[~identified_mask].copy()

                print(f"{len(df_identified)} chunk(s) already identified by BirdNET -> skipping cluster analysis.")
                print(f"{len(df_unidentified)} chunk(s) unidentified -> running cluster analysis on these only.")

                if len(X_unidentified) > 0:
                    # 1. Normalize embeddings (essential for cosine distance)
                    norms = np.linalg.norm(X_unidentified, axis=1, keepdims=True)
                    X_normalized = np.where(norms == 0, X_unidentified, X_unidentified / norms)

                    # 2. Reduce to a moderate-dimensional space FIRST, then cluster on that.
                    print("Reducing embeddings to 10-D with UMAP for clustering...")
                    cluster_reducer = umap.UMAP(
                        n_neighbors=self.umap.n_neighbors,
                        min_dist=self.umap.min_distance,
                        n_components=self.umap.n_components,
                        metric=self.umap.metric,
                        random_state=self.umap.random_state,
                    )
                    X_umap = cluster_reducer.fit_transform(X_normalized)
                    np.save(
                        output_path_results / f"embeddings_8d_{first_file_name}_{analysis_timestamp}_{run_suffix}.npy",
                        X_umap)
                    np.save(
                        output_path_results / f"embeddings_raw_{first_file_name}_{analysis_timestamp}_{run_suffix}.npy",
                        X_normalized)

                    print("Clustering reduced embeddings with HDBSCAN...")
                    clusterer = HDBSCAN(
                        min_cluster_size=self.hdbscan_clusters.min_cluster_size,
                        min_samples=self.hdbscan_clusters.min_samples,
                        metric=self.hdbscan_clusters.cluster_metric,
                        cluster_selection_epsilon=self.hdbscan_clusters.cluster_selection_epsilon,
                    )
                    cluster_labels = clusterer.fit_predict(X_umap)

                    # 3. Separate UMAP run, purely for 2D visualization
                    print("Projecting embeddings to 2D with UMAP for visualization...")
                    viz_reducer = umap.UMAP(
                        n_neighbors=self.umap_second.n_neighbors,
                        min_dist=self.umap_second.min_distance,
                        n_components=self.umap_second.n_components,
                        metric=self.umap_second.metric,
                        random_state=self.umap_second.random_state,
                    )
                    X_2d = viz_reducer.fit_transform(X_normalized)

                    df_unidentified['umap_x'] = X_2d[:, 0]
                    df_unidentified['umap_y'] = X_2d[:, 1]
                    df_unidentified['cluster'] = cluster_labels
                else:
                    print("No unidentified segments to cluster.")

                # Identified rows don't need acoustic clustering — give each species its own "cluster" id.
                if len(df_identified) > 0:
                    df_identified['umap_x'] = np.nan
                    df_identified['umap_y'] = np.nan
                    df_identified['cluster'] = df_identified['birdnet_label'].apply(self.sanitize_for_filename)

                df = pd.concat([df_identified, df_unidentified], ignore_index=True)

                # Save the results
                acoustic_results_path = output_path_results / f"acoustic_clusters_{first_file_name}_{analysis_timestamp}_{run_suffix}.csv"
                df.to_csv(acoustic_results_path, index=False)
                print(f"Clustering complete! Detailed data saved to: {acoustic_results_path}")

                unidentified_clusters = df[(df['birdnet_label'] == "Unidentified/Ambient") & (df['cluster'] != -1)]
                if not unidentified_clusters.empty:
                    print(f"\n[AHA!] Found {len(unidentified_clusters)} unidentified segments that clustered together!")
                    print(unidentified_clusters[['file', 'start_time', 'cluster']].head(10).to_string(index=False))
                else:
                    print("\nNo distinct clusters of unidentified audio found.")
            else:
                print(
                    "\nNo valid embeddings extracted from the audio files. Skipping dimensionality reduction and clustering.")

            if not self.analyze_file_group:
                archive_dir_path = source_audio_dir / "processed" / first_file_name
                archive_dir_path.mkdir(parents=True, exist_ok=True)
                for file_path in audio_files:
                    try:
                        destination = archive_dir_path / file_path.name
                        shutil.move(str(file_path), str(destination))
                        print(f"Moved: {file_path.name} -> processed/{first_file_name}/")
                    except Exception as e:
                        print(f"Failed to move {file_path.name}: {e}")
                # move the log files
                for item in source_audio_dir.glob("*.txt"):
                    shutil.move(item, archive_dir_path)
            else:
                archive_dir_path = source_audio_dir

        # Now extract clusters and species
        run_folder_name = f"{first_file_name}_{analysis_timestamp}_{run_suffix}"
        output_base = output_dir / run_folder_name
        audio_archive = source_audio_dir / "processed" / first_file_name
        cluster_csv = output_base / f"acoustic_clusters_{run_folder_name}.csv"
        out_dir_species = output_base / "species"
        out_dir_clusters = output_base / "clusters"
        os.makedirs(out_dir_species, exist_ok=True)
        os.makedirs(out_dir_clusters, exist_ok=True)

        rows = self.read_rows(csv_path=cluster_csv)
        if not rows:
            sys.exit("No matching rows found in CSV after filtering.")

        # Split rows into species vs ambient datasets
        species_rows = [r for r in rows if not r["is_ambient"]]
        ambient_rows = [r for r in rows if r["is_ambient"]]

        # Group species rows by cluster
        by_cluster_species = defaultdict(list)
        for row in species_rows:
            by_cluster_species[row["cluster"]].append(row)
        for cluster in by_cluster_species:
            by_cluster_species[cluster].sort(key=lambda r: (r["file"], r["start"]))

        # Group ambient rows by cluster
        by_cluster_ambient = defaultdict(list)
        for row in ambient_rows:
            by_cluster_ambient[row["cluster"]].append(row)
        for cluster in by_cluster_ambient:
            by_cluster_ambient[cluster].sort(key=lambda r: (r["file"], r["start"]))

        cache = WavCache(archive_dir_path)

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
                        f"Format mismatch: '{fname}' differs from '{reference_file}' format."
                    )

        framerate = reference_params.framerate
        sampwidth = reference_params.sampwidth
        nchannels = reference_params.nchannels
        silence_frame = b"\x00" * (sampwidth * nchannels)
        gap_frames = int(round((self.gap_ms / 1000.0) * framerate)) if self.gap_ms > 0 else 0
        gap_bytes = silence_frame * gap_frames
        audio_format_params = (nchannels, sampwidth, framerate)

        # Processing Identified Species
        if by_cluster_species:
            print(
                f"\nProcessing {len(species_rows)} identified-species segments across {len(by_cluster_species)} cluster(s)...")
            self.write_cluster_wavs(by_cluster_species, cache, out_dir_species, FILE_PREFIX, MAX_SPECIES_NAME_LENGTH,
                               audio_format_params, gap_bytes)
        else:
            print("\nNo identified-species segments found to export.")

        # Processing Unidentified / Ambient
        if by_cluster_ambient:
            print(
                f"\nProcessing {len(ambient_rows)} unidentified/ambient segments across {len(by_cluster_ambient)} cluster(s)...")
            self.write_cluster_wavs(by_cluster_ambient, cache, out_dir_clusters, FILE_PREFIX, MAX_SPECIES_NAME_LENGTH,
                               audio_format_params, gap_bytes)
        else:
            print("\nNo unidentified/ambient segments found to export.")

        cache.close_all()
        print("\nDone.")
