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
import requests
from mu_utilities.utilities import SQLServerUtilities
from exceptions import RawAudioBatchException
from pinecone import Pinecone

from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import sparse_matrix_sparse_mat_mul

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
    def __init__(self, logger, external_drive, audio_path, output_path, min_confidence, overlap,
                 species_list, gap_ms, hdbscan_clusters, umap, umap_viz, analysis_run_text, analyze_file_group,
                 sqlserver_connection, pinecone_key, birdnet_model_version):
        self.external_drive = external_drive
        self.audio_path = audio_path
        self.output_path = output_path
        self.min_confidence = min_confidence
        self.overlap = overlap
        self.species_list_path = species_list
        self.gap_ms = gap_ms
        self.umap = umap
        self.umap_viz = umap_viz
        self.hdbscan_clusters = hdbscan_clusters
        self.analysis_run_text = analysis_run_text
        self.analyze_file_group = analyze_file_group
        self.sqlserver_connection = sqlserver_connection
        self.pinecone_key = pinecone_key
        self.birdnet_model_version = birdnet_model_version
        BirdNetParserBase.__init__(self, logger=logger)


    def get_regions(self, gps_coordinates):
        lat, lon = gps_coordinates

        # Query Nominatim API with maximum address detail
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=jsonv2&addressdetails=1"
        headers = {"User-Agent": "global_level_extractor_script"}

        try:
            response = requests.get(url, headers=headers).json()
            address = response.get("address", {})

            # Handle village and hamlet combination logic
            village = address.get("village")
            hamlet = address.get("hamlet")

            if village and hamlet:
                local = f"{village}_{hamlet}"
            elif village:
                local = village
            elif hamlet:
                local = hamlet
            else:
                # Fallback to other local district tags if neither village nor hamlet exist
                local = (
                        address.get("quarter")
                        or address.get("suburb")
                        or address.get("neighbourhood")
                        or "N/A"
                )
            # 2. Municipality / Town / City
            municipality = (
                    address.get("town")
                    or address.get("municipality")
                    or address.get("city")
                    or address.get("city_district")
                    or "N/A"
            )

            # 3. Province / State / County
            province = (
                    address.get("province")
                    or address.get("state_district")
                    or address.get("state")
                    or address.get("county")
                    or "N/A"
            )

            # 4. Broader Region / Island Group
            region = (
                    address.get("ISO3166-2-lvl4")
                    or address.get("region")
                    or address.get("state")
                    or "N/A"
            )

            # 5. Country
            country = address.get("country", "N/A")

            return {
                "local": local,
                "municipality": municipality,
                "province": province,
                "region": region,
                "country": country
            }

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

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
                if not is_ambient and self.min_confidence is not None and conf is not None and conf < self.min_confidence:
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
        print(f"  - Skipped {skipped_low_confidence} identified species rows below {self.min_confidence} confidence.")
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

    def _fetch_all_vectors(self, index, namespace=None, filter=None, batch_size=100):
        """
        Pulls all vectors (+ metadata) from a Pinecone namespace.
        """
        all_records = []
        ids_to_fetch = []

        target_namespace = namespace if namespace is not None else ""

        # Step 1: Extract string IDs from ListResponse objects
        for page in index.list(namespace=target_namespace):
            # Handle ListResponse / dict containing 'vectors'
            if hasattr(page, "vectors") and page.vectors:
                # page.vectors is a list of objects/dicts like [{'id': '...'}, ...]
                for vec in page.vectors:
                    if isinstance(vec, dict):
                        ids_to_fetch.append(vec.get("id"))
                    elif hasattr(vec, "id"):
                        ids_to_fetch.append(vec.id)
                    elif isinstance(vec, str):
                        ids_to_fetch.append(vec)
            # Fallback if page directly yields a list/tuple of items
            elif isinstance(page, (list, tuple)):
                for item in page:
                    if isinstance(item, str):
                        ids_to_fetch.append(item)
                    elif isinstance(item, dict) and "id" in item:
                        ids_to_fetch.append(item["id"])
                    elif hasattr(item, "id"):
                        ids_to_fetch.append(item.id)

        # Filter out any None values
        ids_to_fetch = [i for i in ids_to_fetch if i]

        print(f"Found {len(ids_to_fetch)} vector IDs — fetching in batches...")

        if not ids_to_fetch:
            return all_records

        print(f"Sample clean ID: {repr(ids_to_fetch[0])}")  # Confirm clean string output

        # Step 2: Fetch full records in batches
        for i in range(0, len(ids_to_fetch), batch_size):
            batch_ids = ids_to_fetch[i:i + batch_size]
            response = index.fetch(ids=batch_ids, namespace=target_namespace)

            for vec_id, record in response.vectors.items():
                meta = record.metadata or {}
                if filter and not all(meta.get(k) == v for k, v in filter.items()):
                    continue
                all_records.append({
                    "id": vec_id,
                    "values": record.values,
                    "metadata": meta,
                })

        print(f"Retrieved {len(all_records)} vectors after filtering.")
        return all_records

    def extract_embeddings_and_detect(self, file_path, analyzer):
        """
        Runs species detection using the customized species list Analyzer,
        then dynamically builds a patched Analyzer to extract 1024-D embeddings.
        """
        # 1. Standard detection using your custom-list analyzer
        detection_recording = Recording(analyzer=analyzer, path=str(file_path), min_conf=self.min_confidence,
                                        overlap=self.overlap)
        detection_recording.analyze()
        detections = detection_recording.detections

        # 2. Safely swap the interpreter ONLY while creating and running the embedding analyzer
        original_interpreter = tf.lite.Interpreter

        class EmbeddingSafeInterpreter(original_interpreter):
            def __init__(self, *args, **kwargs):
                kwargs['experimental_preserve_all_tensors'] = True
                super().__init__(*args, **kwargs)

        tf.lite.Interpreter = EmbeddingSafeInterpreter

        try:
            embedding_analyzer = Analyzer()
            embedding_overlap = 0.0  # explicit: embeddings pass uses no overlap
            embedding_recording = Recording(analyzer=embedding_analyzer, path=str(file_path), overlap=embedding_overlap)
            embedding_recording.analyze()
            embedding_recording.extract_embeddings()

            raw_embeddings = embedding_recording.embeddings
            chunks = embedding_recording.chunks
        finally:
            tf.lite.Interpreter = original_interpreter

            cleaned_embeddings = []
            for emb in raw_embeddings:
                if emb is None:
                    cleaned_embeddings.append(None)
                    continue

                try:
                    if hasattr(emb, 'numpy'):
                        cleaned_embeddings.append(emb.numpy().flatten())
                    elif isinstance(emb, dict):
                        val = emb.get('array') or emb.get('embeddings') or list(emb.values())[0]
                        cleaned_embeddings.append(np.asarray(val).flatten())
                    elif isinstance(emb, np.ndarray):
                        cleaned_embeddings.append(emb.flatten())
                    else:
                        arr = np.asarray(emb)
                        if arr.size > 0:
                            cleaned_embeddings.append(arr.flatten())
                        else:
                            cleaned_embeddings.append(None)
                except Exception as e:
                    print(f"   [Warning] Failed to parse embedding element: {e}")
                    cleaned_embeddings.append(None)

        chunks_metadata = []

        embedding_step = 3.0 - embedding_overlap  # = 3.0, matches actual embedding chunk spacing

        for i, chunk in enumerate(chunks):
            start_time = i * embedding_step
            end_time = start_time + 3.0

            feat_vector = cleaned_embeddings[i] if i < len(cleaned_embeddings) else None
            if feat_vector is None or feat_vector.shape[0] != 1024:
                continue

            chunk_detections = [
                d for d in detections
                if abs(d['start_time'] - start_time) < 1.5
            ]

            if len(chunk_detections) > 1:
                species_list = ", ".join(d['common_name'] for d in chunk_detections)
                print(f"   [Overlap] Chunk {i} ({start_time:.1f}s-{end_time:.1f}s): "
                      f"{len(chunk_detections)} detections -> {species_list}")

            if chunk_detections:
                labels = [f"{d['common_name']} ({d['scientific_name']})" for d in chunk_detections]
                confidences = [d['confidence'] for d in chunk_detections]
                top_det = max(chunk_detections, key=lambda x: x['confidence'])
            else:
                labels = ["Unidentified/Ambient"]
                confidences = [0.0]
                top_det = None

            chunks_metadata.append({
                "file": file_path.name,
                "start_time": start_time,
                "end_time": end_time,
                "birdnet_labels": labels,  # list of ALL species in this chunk
                "confidences": confidences,  # matching list of confidences
                "birdnet_label": labels[0],  # kept for convenience/back-compat: top or only label
                "confidence": confidences[0] if top_det is None else top_det['confidence'],
                "num_species": len(labels)
            })

        valid_embeddings = [e for e in cleaned_embeddings if e is not None and e.shape[0] == 1024]

        return detections, np.array(valid_embeddings), chunks_metadata

    def extract_and_store(self, source_audio_dir, batch_start, batch_end, index_name="chipbot-birdnet-24"):
        pc = Pinecone(api_key=self.pinecone_key)
        analyzer = Analyzer(custom_species_list_path=self.species_list_path)
        index = pc.Index(index_name)
        # index.delete(delete_all=True, namespace="__default__")

        detection_file_path = source_audio_dir / f"detection_results.txt"

        audio_files = natsort.natsorted(
            [f for f in source_audio_dir.iterdir() if f.suffix.lower() == ".wav"],
            key=lambda x: str(x)
        )
        with open(detection_file_path, "w", encoding="utf-8") as f_out:
            for idx, file_path in enumerate(audio_files, 1):
                print(f"[{idx}/{len(audio_files)}] Embedding: {file_path.name}")
                f_out.write(f"=== File: {file_path.name} ===\n")
                try:
                    detections, embeddings, metadata = self.extract_embeddings_and_detect(
                        file_path, analyzer)
                except Exception as e:
                    print(f"Error on {file_path.name}: {e}")
                    continue

                if len(embeddings) == 0:
                    continue

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
                f_out.write("\n" + "=" * 50 + "\n\n")

                # get all the location metadata from gps coordinates
                my_file_parts = file_path.stem.split("_")
                gps = my_file_parts[2], my_file_parts[3]
                my_device = my_file_parts[0]
                my_datetime = my_file_parts[1]
                utilities = SQLServerUtilities(sp='sp_get_site', sql_server_connection=self.sqlserver_connection,
                                               params_values=gps, params='@lat=?, @long=?', logger=self.logger)
                site = utilities.run_sql_return_params()
                my_site = site[0][0].replace(' ', '-')
                my_locations = self.get_regions(gps)
                my_country = my_locations['country'].replace(' ', '-')
                my_region = my_locations['region'].replace(' ', '-')
                my_province = my_locations['province'].replace(' ', '-')
                my_municipality = my_locations['municipality'].replace(' ', '-')
                my_local = my_locations['local'].replace(' ', '-')

                vectors = []
                for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
                    vectors.append({
                        "id": f"{file_path.stem}_{i}",
                        "values": emb.tolist(),
                        "metadata": {
                            "file": meta["file"],
                            "chunk_start": meta["start_time"],
                            "chunk_end": meta["end_time"],
                            "birdnet_label": meta["birdnet_label"],
                            # top/only species, for simple exact-match filtering
                            "birdnet_labels": meta["birdnet_labels"],  # full list, for $in-style filtering
                            "confidence": meta["confidence"],
                            "num_species": meta["num_species"],
                            "country": my_country,
                            "region": my_region,
                            "province": my_province,
                            "municipality": my_municipality,
                            "local": my_local,
                            "site": my_site,
                            'lat': gps[0],
                            'long': gps[1],
                            'device': my_device,
                            'datetime': my_datetime,
                            "embedding_model_version": self.birdnet_model_version,
                            "min_confidence_used": self.min_confidence,
                            "batch_start": batch_start,
                            "batch_end": batch_end
                        }
                    })
                index.upsert(vectors=vectors)
        print("Extraction and storage complete.")


    def import_file_batch(self):
        self.logger.info("Begin script importing embedding files.")
        audio_extensions = {".wav"}
        audio_files_ext = [f for f in Path(self.external_drive).iterdir() if f.suffix.lower() in audio_extensions]
        if not audio_files_ext:
            msg = f"No matching audio (.wav) files found in external_drive: {self.external_drive}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        audio_files_ext = natsort.natsorted(audio_files_ext, key=lambda x: str(x))
        expected_value = None
        gps = None
        c = 0
        for raw_file in audio_files_ext:
            c += 1
            my_file_parts = raw_file.stem.split("_")
            gps = my_file_parts[2], my_file_parts[3]
            utilities = SQLServerUtilities(sp='sp_get_site', sql_server_connection=self.sqlserver_connection,
                                           params_values=gps, params='@lat=?, @long=?', logger=self.logger)
            current_value = utilities.run_sql_return_params()
            if not current_value:
                msg = f"This file's GPS coordinates cant not be found in an existing site: {raw_file.stem}\n"
                msg += f"GPS coordinates of missing or different site:\n{gps}\n"
                self.logger.error(msg)
                raise RawAudioBatchException(msg)
            if expected_value is None:
                expected_value = current_value
            elif current_value != expected_value:
                # Throw an error immediately if a mismatch occurs
                msg = f"Multiple sites or no site found\n" + str(raw_file) + "\nExpected: {expected_value}\nActual: {current_value}\n"
                msg += f"GPS coordinates of missing or different site:\n{gps}\n"
                self.logger.error(msg)
                raise RawAudioBatchException(msg)

        first_file_name = audio_files_ext[0].stem
        first_file_datetime = first_file_name.split("_")[-3]
        last_file_name = audio_files_ext[-1].stem
        last_file_datetime = last_file_name.split("_")[-3]
        my_locations = self.get_regions(gps)
        my_site = expected_value[0][0].replace(' ', '-')
        my_country = my_locations['country'].replace(' ', '-')
        my_province = my_locations['province'].replace(' ', '-')
        archive_stem = (my_country + "_" + my_province + "_" + my_site + '_' +
                        first_file_datetime + "_" + last_file_datetime)
        archive_dir = Path(self.audio_path) / Path(archive_stem)
        archive_dir.mkdir(parents=True, exist_ok=True)
        for file_path in audio_files_ext:
            if file_path.is_file():
                shutil.move(file_path, archive_dir)
        self.logger.info(str(c) + ' raw files imported into archive directory named: ' + archive_stem)
        c = 0
        log_extensions = {".txt"}
        log_files_ext = [f for f in Path(self.external_drive).iterdir() if f.suffix.lower() in log_extensions]
        for file_path in log_files_ext:
            if file_path.is_file():
                c += 1
                shutil.move(file_path, archive_dir)
        self.logger.info(str(c) + ' log files imported into archive directory named: ' + archive_stem)

        self.extract_and_store(source_audio_dir=archive_dir, batch_start=first_file_datetime,
                               batch_end=last_file_datetime)


    def recluster(self, umap_params, umap_viz_params, hdbscan_params,
                  index_name="chipbot-birdnet-24", only_unidentified=True):

        pc = Pinecone(api_key=self.pinecone_key)  # instantiate once, store as an attribute
        index = pc.Index(index_name)

        desc = pc.describe_index(index_name)  # or index.describe_index_stats() depending on SDK version
        print(desc)

        filter_dict = {'embedding_model_version': self.birdnet_model_version}
        if only_unidentified:
            filter_dict["birdnet_label"] = "Unidentified/Ambient"

        records = self._fetch_all_vectors(index,filter=filter_dict, namespace='__default__')
        if not records:
            print("No matching vectors found.")
            return None
        X = np.array([r["values"] for r in records])
        df = pd.DataFrame([r["metadata"] for r in records])
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = np.where(norms == 0, X, X / norms)
        # Clustering-space UMAP — feeds HDBSCAN
        cluster_reducer = umap.UMAP(
            n_neighbors=self.umap.n_neighbors,
            min_dist=self.umap.min_distance,
            n_components=self.umap.n_components,
            metric=self.umap.metric,
            random_state=self.umap.random_state,
        )
        X_umap = cluster_reducer.fit_transform(X_normalized)
        clusterer = HDBSCAN(
            min_cluster_size=self.hdbscan_clusters.min_cluster_size,
            min_samples=self.hdbscan_clusters.min_samples,
            metric=self.hdbscan_clusters.cluster_metric,
            cluster_selection_epsilon=self.hdbscan_clusters.cluster_selection_epsilon,
        )
        df['cluster'] = clusterer.fit_predict(X_umap)
        df['cluster_probability'] = clusterer.probabilities_
        # Separate viz-space UMAP — independent fit, purely for 2D plotting
        viz_reducer = umap.UMAP(
            n_neighbors=self.umap_viz.n_neighbors,
            min_dist=self.umap_viz.min_distance,
            n_components=self.umap_viz.n_components,
            metric=self.umap_viz.metric,
            random_state=self.umap_viz.random_state,
        )
        X_2d = viz_reducer.fit_transform(X_normalized)
        df['umap_x'] = X_2d[:, 0]
        df['umap_y'] = X_2d[:, 1]
        df.to_csv('output.csv', index=False)
        return df









    def run_pipeline(self):
        # GPU check
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU detected: {gpus}\n")
        else:
            print("No GPU detected. Defaulting to CPU.\n")




        gps = 10.6473347, 124.3891271
        locations = self.get_regions(gps)
        print(locations)
        utilities = SQLServerUtilities(sp='sp_get_site', sql_server_connection=self.sqlserver_connection,
                                       params_values=gps, params='@lat=?, @long=?', logger=self.logger)
        site = utilities.run_sql_return_params()
        print(site)



        if self.analyze_file_group:
            # doing a reanalysis now of existing files
            source_audio_dir = self.audio_path / self.analyze_file_group
            run_suffix = self.analysis_run_text
        else:
            # running new file group fom external drive
            valid_extensions = {".wav", ".txt"}
            for item in Path(self.external_drive).iterdir():
                # Only process non-recursive files matching valid extensions
                if item.is_file() and item.suffix.lower() in valid_extensions:
                    if item.stat().st_size > 0:
                        shutil.move(item, Path(self.audio_path) / item.name)
                    else:
                        item.unlink()

            run_suffix = 'initial'




        analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path_results = Path(Path(self.output_path) / f"{first_file_name}_{analysis_timestamp}_{run_suffix}")
        output_path_results.mkdir(parents=True, exist_ok=True)
        detection_file_path = output_path_results / f"detection_results_{first_file_name}_{analysis_timestamp}_{run_suffix}.txt"

        # Initialize the customized detector
        print("Initializing customized species list analyzer...")
        analyzer = Analyzer(custom_species_list_path=self.species_list_path)



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
                    '''
                    np.save(
                        output_path_results / f"embeddings_8d_{first_file_name}_{analysis_timestamp}_{run_suffix}.npy",
                        X_umap)
                    np.save(
                        output_path_results / f"embeddings_raw_{first_file_name}_{analysis_timestamp}_{run_suffix}.npy",
                        X_normalized)
                    '''
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
                        n_neighbors=self.umap_viz.n_neighbors,
                        min_dist=self.umap_viz.min_distance,
                        n_components=self.umap_viz.n_components,
                        metric=self.umap_viz.metric,
                        random_state=self.umap_viz.random_state,
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
                n_species = df.loc[df['birdnet_label'] != "Unidentified/Ambient", 'birdnet_label'].nunique()
                n_tot_clusters = df.loc[df['cluster'] != -1, 'cluster'].nunique()  # excludes HDBSCAN noise (-1)
                n_clusters = n_tot_clusters - n_species
                total_audio_seconds_approx = len(all_metadata) * 3.0
                noise_rows = df_unidentified[df_unidentified['cluster'] == -1]
                noise_seconds = (noise_rows['end_time'] - noise_rows['start_time']).sum()
                n_noise_chunks = len(noise_rows)
                # Identified species (BirdNET-labeled)
                species_seconds = (df_identified['end_time'] - df_identified['start_time']).sum()
                n_species_chunks = len(df_identified)
                clustered_rows = df_unidentified[df_unidentified['cluster'] != -1]
                clustered_seconds = (clustered_rows['end_time'] - clustered_rows['start_time']).sum()
                n_clustered_chunks = len(clustered_rows)

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
                archive_dir_path = source_audio_dir / first_file_name
                archive_dir_path.mkdir(parents=True, exist_ok=True)
                for file_path in audio_files:
                    try:
                        destination = archive_dir_path / file_path.name
                        shutil.move(str(file_path), str(destination))
                    except Exception as e:
                        print(f"Failed to move {file_path.name}: {e}")
                # move the log files
                for item in source_audio_dir.glob("*.txt"):
                    shutil.move(item, archive_dir_path)
            else:
                archive_dir_path = source_audio_dir

        # Now extract clusters and species
        output_dir = Path(self.output_path)
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

        with open(output_path_results / "summary.txt", "a") as summary:
            summary.write(str(total_audio_seconds_approx) + ' total seconds of audio analyzed\n')
            summary.write(str(n_species) + ' species detected in ' + str(species_seconds) + ' seconds of audio\n')
            summary.write(str(n_clusters) + ' additional unique clusters in ' + str(clustered_seconds) + ' seconds of audio\n')
            summary.write(str(noise_seconds) + ' seconds of background noise\n\n')
            summary.write('Species Detection Min confidence: ' + str(self.min_confidence) + '\n\n')
            summary.write('Overlap: ' + str(self.overlap) + '\n\n')
            summary.write('HDBSCAN Parameters used\n')
            summary.write('Min cluster size: ' + str(self.hdbscan_clusters.min_cluster_size) + '\n')
            summary.write('Min samples: ' + str(self.hdbscan_clusters.min_samples) + '\n')
            summary.write('Metric: ' + str(self.hdbscan_clusters.cluster_metric) + '\n')
            summary.write('Selection Epsilon: ' + str(self.hdbscan_clusters.cluster_selection_epsilon) + '\n')
        summary.close()


        print("\nDone.")
