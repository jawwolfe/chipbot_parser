import numpy as np
import pandas as pd
from pathlib import Path
import natsort
import tensorflow as tf
# Clustering and reduction
import umap
from sklearn.cluster import HDBSCAN
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import shutil, os, csv, sys, wave, re, json
from datetime import datetime, timedelta, timezone
import requests
from mu_utilities.utilities import SQLServerUtilities
from mu_utilities.exceptions import DatabaseIntegrityException
from exceptions import RawAudioBatchException, VerifyFileException
from pinecone import Pinecone
from operator import itemgetter
import soundfile as sf
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import sparse_matrix_sparse_mat_mul
from itertools import groupby
from operator import attrgetter
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from functools import partial

IGNORED_LABEL = "unidentified/ambient"
DEFAULT_LAT = '10.6608806'
DEFAULT_LONG = '124.3595418'
CLUSTER_ALGORITHM = 'HDBSCAN'

@dataclass
class ClusterChunkRow:
    chunk_id: str
    run_id: int
    cluster_id: int
    cluster_probability: float
    file_stem: str
    chunk_index: int
    abs_start_ms: int
    abs_end_ms: int
    start_datetime: datetime
    end_datetime: datetime
    directory: str
    device: str
    start_sample: int
    end_sample: int

@dataclass
class DetectionRow:
    detection_id: int
    chunk_id: str
    model_id: int
    species: str
    confidence: float
    start_sample: int
    end_sample: int
    file_stem: str
    chunk_index: int
    abs_start_ms: int
    abs_end_ms: int
    start_datetime: datetime
    end_datetime: datetime
    directory: str
    device: str

class SegmentType(Enum):
    DETECTION = "detection"
    CLUSTER = "cluster"

@dataclass
class Segment:
    segment_type: SegmentType
    file_stem: str
    first_abs_start_ms: int
    last_abs_end_ms: int
    first_datetime: datetime
    last_datetime: datetime
    first_chunk_id: str
    last_chunk_id: str
    chunk_ids: list
    directory: str
    device: str
    members: list  # list[DetectionRow] or list[ClusterChunkRow], depending on segment_type
    # detection-only
    species: str | None = None
    # cluster-only
    run_id: int | None = None
    cluster_id: int | None = None
    avg_cluster_probability: float | None = None
    segment_id: int | None = None

class BirdNetParserBase:
    def __init__(self, logger):
        self.logger = logger

class WavCache:
    """Keeps source WAV files open (read-only) and caches their audio params."""

    def __init__(self):
        self._handles = {}
        self._params = {}

    def get(self, directory, file_stem):
        key = (directory, file_stem)
        if key not in self._handles:
            path = Path(directory) / f"{file_stem}.wav"
            if not path.is_file():
                raise FileNotFoundError(f"Source WAV not found: {path}")
            wf = wave.open(str(path), "rb")
            self._handles[key] = wf
            self._params[key] = wf.getparams()
        return self._handles[key]

    def params(self, directory, file_stem):
        self.get(directory, file_stem)
        return self._params[(directory, file_stem)]

    def close_all(self):
        for wf in self._handles.values():
            wf.close()

class BirdNetParser(BirdNetParserBase):
    def __init__(self, logger, external_drive, audio_path, cluster_links, detection_links,clips_path, min_confidence,
                 overlap, species_list, gap_tolerance_ms, hdbscan_clusters, umap, umap_viz, analysis_run_text,
                 analyze_file_group, sqlserver_connection, pinecone_key, birdnet_model_version, min_cluster_probability):
        self.external_drive = external_drive
        self.audio_path = audio_path
        self.clips_path = clips_path
        self.cluster_links = cluster_links
        self.detection_links = detection_links
        self.species_list_path = species_list
        self.min_confidence = min_confidence
        self.overlap = overlap
        self.gap_tolerance_ms = gap_tolerance_ms
        self.min_cluster_probability = min_cluster_probability
        self.umap = umap
        self.umap_viz = umap_viz
        self.hdbscan_clusters = hdbscan_clusters
        self.analysis_run_text = analysis_run_text
        self.analyze_file_group = analyze_file_group
        self.sqlserver_connection = sqlserver_connection
        self.pinecone_key = pinecone_key
        self.birdnet_model_version = birdnet_model_version
        BirdNetParserBase.__init__(self, logger=logger)

    def parse_log(self, path):
        with open(path, "r") as f:
            text = f.read()
        lines = text.splitlines()
        entries = []
        last_reading = None
        temp_re = re.compile(
            r"Temp:\s*([\-\d.]+)\s*°C\s*\|\s*Hum:\s*([\-\d.]+)\s*%\s*\|\s*Bat:\s*([\-\d.]+)"
        )
        file_re = re.compile(r"^/?([a-zA-Z]{2}-chipbot-[^\s]+)\.wav")
        for line in lines:
            line = line.strip()
            m = temp_re.search(line)
            if m:
                last_reading = {
                    "Temp": m.group(1),
                    "Hum": m.group(2),
                    "Bat": m.group(3),
                }
                continue
            m = file_re.match(line)
            if m and last_reading is not None:
                filename = m.group(1)
                entries.append({
                    "Filename": filename,
                    "Temp": last_reading["Temp"],
                    "Hum": last_reading["Hum"],
                    "Bat": last_reading["Bat"],
                })
        logfilename = os.path.basename(path)
        # Return as a list containing the single log entry dictionary
        return [{"logfilename": logfilename, "data": entries}]

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

    def get_wav_duration(self, file_path: str | Path) -> float:
        path_obj = Path(file_path)

        if not path_obj.exists() or path_obj.stat().st_size == 0:
            return 0.0

        try:
            info = sf.info(str(path_obj))
            return float(info.duration)
        except Exception:
            # Handles sf.LibsndfileError, corrupt headers, or unrecognized formats
            return 0.0

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
        raw_embeddings = []
        embedding_overlap = None
        chunks= None
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
                    msg = f"[Warning] Failed to parse embedding element: {e}"
                    self.logger.error(msg)
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


    def get_abs_chunks_datetime(self, filename, start_sample, end_sample):
        datetime_str = filename.split("_")[1]
        base_dt = datetime.strptime(datetime_str, "%Y-%m-%d-%H%M%S")
        abs_start = base_dt + timedelta(seconds=float(start_sample))
        abs_end = base_dt + timedelta(seconds=float(end_sample))
        return_value = (abs_start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], abs_end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        return return_value


    def extract_and_store(self, source_audio_dir, index_name="chipbot-birdnet-24"):
        #pc = Pinecone(api_key=self.pinecone_key)
        #index = pc.Index(index_name)
        # index.delete(delete_all=True, namespace="__default__")
        analyzer = Analyzer(custom_species_list_path=self.species_list_path)
        audio_files = natsort.natsorted(
            [f for f in source_audio_dir.iterdir() if f.suffix.lower() == ".wav"],
            key=lambda x: str(x)
        )

        for idx, file_path in enumerate(audio_files, 1):
            try:
                detections, embeddings, metadata = self.extract_embeddings_and_detect(file_path, analyzer)
            except Exception as e:
                print(f"Error on {file_path.name}: {e}")
                continue
            i = 0
            insert_data_chunk = []
            insert_data_embed = []
            insert_data_detect = []
            for item_m, item_e in zip(metadata, embeddings):
                chunk_id_m = file_path.stem + "_" + str(int(item_m['start_time'] / 3))
                vector_str = json.dumps(item_e.tolist())
                values = self.get_abs_chunks_datetime(item_m['file'][:-4], item_m['start_time'], item_m['end_time'])
                insert_data_chunk.append((chunk_id_m, item_m['file'][:-4], item_m['start_time'], item_m['end_time'], i,
                                          values[0], values[1]))
                insert_data_embed.append((chunk_id_m, float(self.birdnet_model_version), vector_str))
                i += 1
            for item_d in detections:
                chunk_id = file_path.stem + "_" + str(int(item_d['start_time'] / 3))
                insert_data_detect.append((chunk_id, float(self.birdnet_model_version),
                                           item_d['common_name'] + ' (' + item_d['scientific_name'] + ')',
                                           item_d['confidence']))
                
            insert_sql = ("INSERT INTO Chunks (ChunkID, FileName, StartSample, EndSample, ChunkIndex, AbsStartMS, "
                          "AbsEndMS) VALUES (?, ?, ?, ?, ?, ?, ?)")
            utilities = SQLServerUtilities(sql=insert_sql, sql_server_connection=self.sqlserver_connection,
                                           params_values=insert_data_chunk, logger=self.logger)
            try:
                self.logger.info(f"{str(len(insert_data_chunk))} chunks.")
                utilities.run_plain_sql_bulk_params()
            except DatabaseIntegrityException as err:
                first_value = insert_data_chunk[0]
                msg = f"Duplicates in the chunks batch commit rolled back.{str(first_value)}"
                self.logger.error(msg)

            insert_sql = "INSERT INTO ChunkEmbeddings (ChunkID, ModelID, VectorBlob) VALUES (?, ?, ?)"
            utilities = SQLServerUtilities(sql=insert_sql, sql_server_connection=self.sqlserver_connection,
                                           params_values=insert_data_embed, logger=self.logger)
            try:
                self.logger.info(f"{str(len(insert_data_embed))} embeddings.")
                utilities.run_plain_sql_bulk_params()
            except DatabaseIntegrityException as err:
                first_value = insert_data_embed[0]
                msg = f"Duplicates in the chunk embeddings batch commit rolled back."
                self.logger.error(msg)

            if insert_data_detect:
                insert_sql = "INSERT INTO Detections (ChunkID, ModelID, Species, Confidence) VALUES (?, ?, ?, ?)"
                utilities = SQLServerUtilities(sql=insert_sql, sql_server_connection=self.sqlserver_connection,
                                               params_values=insert_data_detect, logger=self.logger)
                try:
                    self.logger.info(f"{str(len(insert_data_detect))} detections.")
                    utilities.run_plain_sql_bulk_params()
                except DatabaseIntegrityException as err:
                    first_value = insert_data_detect[0]
                    msg = f"Duplicates in the detections batch commit rolled back.{str(first_value)}"
                    self.logger.error(msg)

        self.logger.info("This batch extraction and storage complete.")


    def import_file_batch(self):
        self.logger.info("Begin script importing embedding files.")
        log_extensions = {".txt"}
        log_files_ext = [f for f in Path(self.external_drive).iterdir() if f.suffix.lower() in log_extensions]
        log_files_ext_data = []
        for log_file in log_files_ext:
            log_files_ext_data.extend(self.parse_log(log_file))
        log_files_ext_data.sort(key=lambda entry: entry["logfilename"])
        audio_extensions = {".wav"}
        audio_files_ext = [f for f in Path(self.external_drive).iterdir() if f.suffix.lower() in audio_extensions]
        if not audio_files_ext:
            msg = f"No matching audio (.wav) files found in external_drive: {self.external_drive}"
            self.logger.error(msg)
            raise VerifyFileException(msg)

        audio_files_ext = natsort.natsorted(audio_files_ext, key=lambda x: str(x))
        # quality check the log file against all the audio files in the directory
        for item_file in audio_files_ext:
            flag = False
            for batch in log_files_ext_data:
                for item_log in batch['data']:
                    if item_log['Filename'] == item_file.stem:
                        flag = True
            if not flag:
                msg = f"Audio file in log missing from external drive {item_file.stem}"
                self.logger.error(msg)
                raise VerifyFileException(msg)

        for batch in log_files_ext_data:
            # note that all files in a batch (one log file) have same gps coordinates first is same as all files
            # so here we handle site, location and batch at this level
            c = 0
            sorted_data = sorted(batch['data'], key=lambda x: x['Filename'])
            first_dict = sorted_data[0]
            last_dict = sorted_data[-1]
            first_timestamp = first_dict['Filename'].split('_')[-3]
            last_timestamp = last_dict['Filename'].split('_')[-3]
            my_file_parts = first_dict['Filename'].split("_")
            gps = my_file_parts[2], my_file_parts[3]
            if gps == ('0.000000', '0.000000'):
                msg = f"This file's GPS coordinates are not known: {first_dict['Filename']}. Attempt to use hard coded.\n"
                self.logger.error(msg)
                if DEFAULT_LAT != '0.000000' and DEFAULT_LONG != '0.000000':
                    gps = (DEFAULT_LAT, DEFAULT_LONG)
                for old_file_name in batch['data']:
                    old_name_path = self.external_drive / Path(old_file_name['Filename'] + ".wav")
                    new_filename = old_name_path.name.replace("0.000000", gps[0], 1).replace(
                        "0.000000", gps[1], 1)
                    new_name_path = old_name_path.with_name(new_filename)
                    if old_name_path.exists():
                        shutil.move(old_name_path, new_name_path)

            utilities = SQLServerUtilities(sp='sp_get_site_by_coordinates',
                                           sql_server_connection=self.sqlserver_connection,
                                           params_values=gps, params='@Latitude=?, @Longitude=?', logger=self.logger)
            site_data = utilities.run_sql_return_params()
            if not site_data:
                msg = f"This file's GPS coordinates cant not be found in an existing site: {first_dict['Filename']}\n"
                self.logger.error(msg)
                raise RawAudioBatchException(msg)

            my_locations = self.get_regions(gps)
            location_params = (my_locations['country'], my_locations['region'], my_locations['province'],
                               my_locations['municipality'], my_locations['local'], site_data[0][0])
            utilities = SQLServerUtilities(sp='sp_get_insert_location',
                                           sql_server_connection=self.sqlserver_connection,
                                           params_values=location_params, params='@Level1=?, @Level2=?, @Level3=?, '
                                                                                 '@Level4=?, @Level5=?, @SiteID=?',
                                           logger=self.logger)
            location_id = utilities.run_sql_return_params()[0][0]
            lon = gps[1]
            lat = gps[0]
            gps_wkt = f"POINT({lon} {lat})"
            my_site_name = site_data[0][1].replace(' ', '-')
            my_country = my_locations['country'].replace(' ', '-')
            my_province = my_locations['province'].replace(' ', '-')
            archive_stem = (my_country + "_" + my_province + "_" + my_site_name + '_' +
                            first_timestamp + "_" + last_timestamp)
            batch_params = (my_file_parts[0], gps_wkt, location_id, first_timestamp, last_timestamp, archive_stem)
            utilities = SQLServerUtilities(sp='sp_get_insert_batch', sql_server_connection=self.sqlserver_connection,
                                           params_values=batch_params, params='@DeviceName=?, @GpsCoordinatesText=?, '
                                                                              '@LocationID=?, @BatchStart=?, '
                                                                              '@BatchEnd=?, @Directory=?',
                                           logger=self.logger)
            batch_id = utilities.run_sql_return_params()[0][0]
            self.logger.info(f"Processing {len(batch['data'])} files in batch: {my_file_parts[0]}_{first_timestamp}_{last_timestamp}_{lat}_{lon}")
            archive_dir = Path(self.audio_path) / Path(archive_stem)
            archive_dir.mkdir(parents=True, exist_ok=True)

            for log_file in batch['data']:
                c += 1
                # handle case when gps coordinates where 0.000000 then replaced with defaults
                new_filename = log_file['Filename'].replace("0.000000", gps[0], 1).replace(
                    "0.000000", gps[1], 1)
                wav_file_path = Path(self.external_drive) / Path(new_filename + ".wav")
                file_split = wav_file_path.stem.split("_")
                file_length = self.get_wav_duration(wav_file_path)
                file_params = (batch_id, new_filename, log_file['Temp'], log_file['Hum'], log_file['Bat'],
                               file_split[1], file_length)
                utilities = SQLServerUtilities(sp='sp_get_insert_file',
                                               sql_server_connection=self.sqlserver_connection,
                                               params_values=file_params, params='@BatchID=?, @FileFullName=?, '
                                                                                  '@Temp=?, @Humidity=?, @Battery=?, '
                                                                                  '@DatetimeStart=?, @Length=?',
                                               logger=self.logger)
                self.logger.info(f"Processing File: {new_filename}")
                if file_length == 0.0:
                    # delete file
                    wav_file_path.unlink(missing_ok=True)
                else:
                    file_id = utilities.run_sql_return_params()[0][0]
                    # move file to archive
                    shutil.move(wav_file_path, archive_dir)
            log_file_path = self.external_drive / Path(batch['logfilename'])
            shutil.move(self.external_drive / log_file_path, archive_dir)
            self.logger.info("This batch files have been processed and moved to archive.")
            self.logger.info("Begin Birdnet Embedding and Database insert.")
            # now insert all embeddings for this batch into SQL Server
            self.extract_and_store(source_audio_dir=archive_dir)
            detections = self.fetch_all_detections(batch_id)
            segments_detections = self.build_detection_segments(detections, gap_tolerance_ms=self.gap_tolerance_ms)
            clips_root = Path(self.clips_path)
            detection_out = clips_root
            written_detections = self.carve_segment_clips(
                segments_detections,
                detection_out,
                make_relpath=partial(self._detection_clip_relpath, sanitize_species=self.sanitize_for_filename),
            )
            self.build_detection_link_tree(written_detections, links_root=self.detection_links)
            utilities = SQLServerUtilities(sp='sp_get_batch_metrics',
                                           sql_server_connection=self.sqlserver_connection, params_values=batch_id,
                                           params='@BatchID=?', logger=self.logger)
            data_stats = utilities.run_sql_return_params()[0]
            run_path = Path(self.detection_links) / Path(archive_stem) / Path(f"summary_batch_{batch_id}.txt")
            with open(run_path, "a") as summary:
                summary.write(f'Detections Summary: \n')
                summary.write(str(data_stats[0]) + ' 3 second chunks with detections.\n')
                summary.write(str(data_stats[1]) + ' species identified.\n')
                summary.write(str(data_stats[2]) + ' species segments collected.\n\n')
                summary.write('Detection Parameters used:\n')
                summary.write(str(self.min_confidence) + ' minimum confidence required.\n')
                summary.write(str(self.overlap) + ' overlap (seconds).\n\n')
                summary.write('Segmentation Parameters used:\n')
                summary.write(str(self.gap_tolerance_ms / 1000) + ' gap tolerance (seconds).\n\n')
            summary.close()

    def process(self):
        #self.extract_and_store(source_audio_dir=Path('C:\\temp\\CHIPBOT_DATA_ROOT\\input\\United-States_Indiana_Indianapolis-House-Backyard_2026-07-18-053107_2026-07-18-082620'))
        detections = self.fetch_all_detections(12)
        segments_detections = self.build_detection_segments(detections, gap_tolerance_ms=self.gap_tolerance_ms)
        clips_root = Path(self.clips_path)
        detection_out = clips_root
        written_detections_1 = self.carve_segment_clips(
            segments_detections,
            detection_out,
            make_relpath=partial(self._detection_clip_relpath, sanitize_species=self.sanitize_for_filename),
        )
        self.build_detection_link_tree(written_detections_1, links_root=self.detection_links)
        utilities = SQLServerUtilities(sp='sp_get_batch_metrics',
                                          sql_server_connection=self.sqlserver_connection, params_values=14,
                                       params='@BatchID=?', logger=self.logger)
        data_stats = utilities.run_sql_return_params()[0]
        run_path = (Path(self.detection_links)
                    / Path(f"Philippines_Cebu_Pacijan-Lake-Danao-Marsh-East_2026-09-03-111353_2026-09-03-125900")
                    / Path(f"summary_batch_14.txt"))
        with open(run_path, "a") as summary:
            summary.write(f'Detections Summary: \n')
            summary.write(str(data_stats[0]) + ' 3 second chunks with detections.\n')
            summary.write(str(data_stats[1]) + ' species identified.\n')
            summary.write(str(data_stats[2]) + ' species segments collected.\n\n')
            summary.write('Detection Parameters used:\n')
            summary.write(str(self.min_confidence) + ' minimum confidence required.\n')
            summary.write(str(self.overlap) + ' overlap (seconds).\n\n')
            summary.write('Segmentation Parameters used:\n')
            summary.write(str(self.gap_tolerance_ms / 1000) + ' gap tolerance (seconds).\n\n')
        summary.close()

    def clusterer(self, batch=None, only_unidentified=True, start_date=None, end_date=None,
                  site_list = None, level_1 = None, level_2 = None, level_3 = None):

        jconfig_payload = {
        "hdbscan": self.hdbscan_clusters,
        "umap": self.umap,
        "umap_viz": self.umap_viz
        }
        clean_ml_parameters = {
            "hdbscan": vars(self.hdbscan_clusters),
            "umap": vars(self.umap),
            "umap_viz": vars(self.umap_viz),
        }

        params_json_payload = json.dumps(jconfig_payload, default=vars)

        sql='Select c.[ChunkID], ce.VectorBlob '
        sql+='from Chunks c '
        sql+='inner join ChunkEmbeddings ce on c.ChunkID = ce.ChunkID '
        sql+='inner join Files f on FileFullName = c.[FileName] '
        sql+='left outer join Detections d on d.ChunkID = c.ChunkID '
        sql+='inner join [Batches] b on b.BatchID = f.BatchID '
        sql+='inner join Locations l on l.LocationID = b.LocationID '
        sql+='where d.DetectionID '
        if only_unidentified:
            sql += 'is null '
        else:
            sql += 'is not null '
        if batch:
            sql += "and b.BatchID = '" + str(batch) + "' "
        else:
            if site_list:
                formatted_sites = ",".join(map(str, site_list))
                sql += f"and l.SiteID in ({formatted_sites}) "
            if level_1:
                sql += "and l.Level1 in = '" + level_1 + "' "
            if level_2:
                sql += "and l.Level2 in = '" + level_2 + "' "
            if level_3:
                sql += "and l.Level3 in = '" + level_3 + "' "
            if start_date and end_date:
                sql += "and f.DatetimeStart between '" + start_date + "' and '" + end_date + "' "

        utilities = SQLServerUtilities(sql=sql, sql_server_connection=self.sqlserver_connection,
                                       logger=self.logger)

        records = utilities.run_plain_sql_return()
        chunk_ids = [r[0] for r in records]
        parsed_rows = [
            json.loads(r[1]) if isinstance(r[1], str) else r[1]
            for r in records
        ]
        X = np.array(parsed_rows, dtype=float)
        df = pd.DataFrame()
        df['ChunkID'] = chunk_ids
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
        counts_dict = df["cluster"].value_counts().to_dict()

        # enter run
        run_params = (CLUSTER_ALGORITHM, params_json_payload, float(self.birdnet_model_version), only_unidentified,
                      len(records), start_date, end_date, level_1, level_2, level_3, batch, "", None)
        utilities = SQLServerUtilities(sp='sp_insert_run',
                                       sql_server_connection=self.sqlserver_connection, params_values=run_params,
                                       params='@Algorithm=?, @Parameters=?, @ModelID=?, @Unidentified=?, @ChunkCount=?, '
                                              '@StartDate=?, @EndDate=?, @LocationLevel1=?, @LocationLevel2=?, '
                                              '@LocationLevel3=?, @BatchID=?, @Notes=?, @NewRunID=? OUTPUT', logger=self.logger)
        run_id = utilities.run_sql_return_params(result_scalar=True)[0]
        df['RunID'] = run_id

        # enter site runs if needed
        if site_list:
            for site in site_list:
                run_params = (run_id, site)
                utilities = SQLServerUtilities(sp='sp_insert_run_site',
                                               sql_server_connection=self.sqlserver_connection, params_values=run_params,
                                               params='@RunID=?, @SiteID=?', logger=self.logger)
                utilities.run_sql_params()

        # enter clusters
        centroids = {
            cluster_num: X_normalized[df['cluster'].values == cluster_num].mean(axis=0)
            for cluster_num in counts_dict
        }

        for cluster_num, row_count in counts_dict.items():
            centroid_blob = json.dumps(centroids[cluster_num].tolist())
            run_params = (run_id, cluster_num, row_count, centroid_blob, None)
            utilities = SQLServerUtilities(sp='sp_insert_cluster', sql_server_connection=self.sqlserver_connection,
                                           params_values=run_params, params='@RunID=?, @ClusterID=?, @Size=?, '
                                                                            '@CentroidBlob=?, @Label=?',
                                           logger=self.logger)
            utilities.run_sql_params()

        # enter cluster chunks
        cols_in_order = ['ChunkID', 'cluster', 'cluster_probability', 'umap_x', 'umap_y', 'RunID']
        data_to_insert = [tuple(row) for row in df[cols_in_order].itertuples(index=False, name=None)]
        utilities = SQLServerUtilities(sp='sp_insert_cluster_chunks', sql_server_connection=self.sqlserver_connection,
                                       params_values=data_to_insert, params='@ChunkID=?, @ClusterID=?, '
                                                                            '@ClusterProbability=?, @UmapX=?, @UmapY=?, '
                                                                            '@RunID=?', logger=self.logger)
        utilities.run_sql_bulk_params()
        segments = self.run_cluster_segmentation(run_id)

        clips_root = Path(self.clips_path)
        cluster_out = clips_root
        # todo create a file with parameters and location/sites/dates
        #cluster_out.mkdir(parents=True, exist_ok=True)

        written_cluster_clips = self.carve_segment_clips(
            segments,
            cluster_out,
            make_relpath=self._cluster_clip_relpath,
        )
        self.build_cluster_link_tree(written_cluster_clips, links_root=self.cluster_links)
        utilities = SQLServerUtilities(sp='sp_get_run_metrics',
                                          sql_server_connection=self.sqlserver_connection, params_values=run_id,
                                       params='@RunID=?', logger=self.logger)
        data_stats = utilities.run_sql_return_params()
        statistics = dict(data_stats)
        chunks = statistics['Total Chunk Count']
        run_path = Path(self.cluster_links) / f"run_{run_id}" / Path(f"summary_run{run_id}.txt")
        with open(run_path, "a") as summary:
            summary.write('Clusters Query: \n')
            summary.write('Only Unidentified: ' + str(only_unidentified) + '\n')
            summary.write(f'Batch: {batch}\n')
            summary.write(f'Level1: {level_1}\n')
            summary.write(f'Level2: {level_2}\n')
            summary.write(f'Level3: {level_3}\n')
            summary.write(f'Site List: {site_list}\n')
            summary.write(f'Start Date: {start_date}\n')
            summary.write(f'End Date: {end_date}\n\n')
            summary.write(f'Clusters Summary: \n')
            summary.write(str(statistics['Total Duration']) + ' minutes of audio analyzed.\n')
            summary.write(str(statistics['Noise Duration']) + ' minutes of noise in cluster -1.\n')
            summary.write(str(statistics['Clustered Duration']) + ' minutes of audio clustered.\n')
            summary.write(str(statistics['Total Chunk Count']) + ' 3 second chunks.\n')
            summary.write(str(statistics['Total Cluster Count']) + ' clusters identified.\n')
            summary.write(str(statistics['Total Segment Count']) + ' segments collected.\n\n')
            summary.write('Segmentation Parameters used:\n')
            summary.write(str(self.min_cluster_probability) + ' min cluster probability.\n')
            summary.write(str(self.gap_tolerance_ms / 1000) + ' gap tolerance (seconds).\n\n')
            summary.write('HDBSCAN and NMAP Parameters used:\n')
            json.dump(clean_ml_parameters, summary, indent=4)
        summary.close()



    def parse_chunk_id(self, chunk_id: str) -> tuple[str, int]:
        file_stem, _, index_str = chunk_id.rpartition("_")
        return file_stem, int(index_str)


    def run_cluster_segmentation(self, run_id):
        params = (run_id, self.gap_tolerance_ms, self.min_cluster_probability)
        utilities = SQLServerUtilities(sp='sp_get_cluster_segment_data',
                                       sql_server_connection=self.sqlserver_connection,
                                       params_values=params,
                                       params='@RunID=?, @GapToleranceMS=?, @MinClusterProbability=?',
                                       logger=self.logger)
        rows = utilities.run_sql_return_params()

        columns = ['RunID', 'ClusterID', 'ChunkID', 'ClusterProbability', 'DeviceName',
                   'FileName', 'ChunkIndex', 'AbsStartMS', 'AbsEndMS', 'SegmentGroupID', 'Directory', 'StartSample',
                   'EndSample']
        df = pd.DataFrame.from_records(rows, columns=columns)

        segments = []
        for (grp_run_id, cluster_id, segment_group_id), group_df in df.groupby(
                ['RunID', 'ClusterID', 'SegmentGroupID']):
            segment_id = None
            group_df = group_df.sort_values('AbsStartMS')

            members = [
                ClusterChunkRow(
                    chunk_id=row.ChunkID,
                    run_id=row.RunID,
                    cluster_id=row.ClusterID,
                    cluster_probability=row.ClusterProbability,
                    file_stem=row.FileName,
                    chunk_index=row.ChunkIndex,
                    abs_start_ms=int(row.AbsStartMS.timestamp() * 1000),
                    abs_end_ms=int(row.AbsEndMS.timestamp() * 1000),
                    start_datetime=row.AbsStartMS,
                    end_datetime=row.AbsEndMS,
                    directory=row.Directory,
                    device=row.DeviceName,
                    start_sample=row.StartSample,
                    end_sample=row.EndSample
                )
                for row in group_df.itertuples()
            ]

            segments.append(Segment(
                segment_type=SegmentType.CLUSTER,
                file_stem=members[0].file_stem,
                first_abs_start_ms=members[0].abs_start_ms,
                last_abs_end_ms=members[-1].abs_end_ms,
                first_datetime=members[0].start_datetime,
                last_datetime=members[-1].end_datetime,
                first_chunk_id=members[0].chunk_id,
                last_chunk_id=members[-1].chunk_id,
                chunk_ids=[m.chunk_id for m in members],
                directory=members[0].directory,
                device=members[0].device,
                members=members,
                run_id=grp_run_id,
                cluster_id=cluster_id,
                avg_cluster_probability=group_df['ClusterProbability'].mean(),
                segment_id=segment_id
            ))
        for segment in segments:
            segment.segment_id = self._commit_segment(segment)
        return segments


    def fetch_all_detections(self, batch_id) -> list[DetectionRow]:
        params = (batch_id)
        utilities = SQLServerUtilities(sp='sp_get_detections_by_batch', sql_server_connection=self.sqlserver_connection,
                                       params_values=params, params='@BatchID=?', logger=self.logger)
        detections = utilities.run_sql_return_params()
        rows = []

        for item in detections:
            file_stem, chunk_index = self.parse_chunk_id(item[1])
            abs_start_ms = int(item[8].replace(tzinfo=timezone.utc).timestamp() * 1000)
            abs_end_ms = int(item[9].replace(tzinfo=timezone.utc).timestamp() * 1000)
            rows.append(DetectionRow(
                detection_id=item[0], chunk_id=item[1], model_id=item[2], species=item[3], confidence=item[4],
                start_sample=item[5], end_sample=item[6], file_stem=file_stem, chunk_index=chunk_index,
                abs_start_ms=abs_start_ms, abs_end_ms=abs_end_ms, directory=item[7], start_datetime=item[8],
                end_datetime=item[9], device=item[10]
            ))
        rows.sort(key=lambda r: (r.abs_start_ms))
        return rows

    def _commit_segment(self, current: Segment) -> None:
        run_params = (current.device, current.first_datetime, current.last_datetime, len(current.chunk_ids), None)
        utilities = SQLServerUtilities(sp='sp_insert_segment',
                                       sql_server_connection=self.sqlserver_connection,
                                       params_values=run_params,
                                       params='@DeviceName=?, @StartTime=?, @EndTime=?, @ChunkCount=?, '
                                              '@NewSegmentID=? OUTPUT', logger=self.logger)
        try:
            segment_id = utilities.run_sql_return_params()[0][0]
            is_new_segment = True
        except DatabaseIntegrityException:
            self.logger.error("Duplicate segment commit rolled back.")
            run_params = (current.device, current.first_datetime, current.last_datetime)
            utilities = SQLServerUtilities(sp='sp_get_segment_id',
                                           sql_server_connection=self.sqlserver_connection,
                                           params_values=run_params,
                                           params='@DeviceName=?, @StartTime=?, @EndTime=?', logger=self.logger)
            segment_id = utilities.run_sql_return_params()[0][0]
            is_new_segment = False

        if segment_id is None:
            self.logger.error("Could not resolve segment_id for duplicate segment; skipping.")
            return None

        if is_new_segment:
            for index, value in enumerate(current.chunk_ids):
                my_params = (segment_id, value, index)
                utilities = SQLServerUtilities(sp='sp_insert_segment_chunk',
                                               sql_server_connection=self.sqlserver_connection,
                                               params_values=my_params,
                                               params='@SegmentID=?, @ChunkID=?, @Position=?', logger=self.logger)
                utilities.run_sql_params()

        if current.segment_type == SegmentType.DETECTION:
            detection_id = current.members[0].detection_id
            my_params = (detection_id, segment_id)
            utilities = SQLServerUtilities(sp='sp_insert_segment_detection',
                                           sql_server_connection=self.sqlserver_connection,
                                           params_values=my_params,
                                           params='@DetectionID=?, @SegmentID=?', logger=self.logger)
            try:
                utilities.run_sql_params()
            except DatabaseIntegrityException:
                self.logger.error("Duplicate segment detection commit rolled back.")

        if current.segment_type == SegmentType.CLUSTER:
            cluster_id = current.members[0].cluster_id
            my_params = (cluster_id, segment_id, current.run_id, current.avg_cluster_probability)
            utilities = SQLServerUtilities(sp='sp_insert_segment_cluster',
                                           sql_server_connection=self.sqlserver_connection,
                                           params_values=my_params,
                                           params='@ClusterID=?, @SegmentID=?, @RunID=?, @MeanClusterProbability=?',
                                           logger=self.logger)
            try:
                utilities.run_sql_params()
            except DatabaseIntegrityException:
                self.logger.error("Duplicate cluster segment commit rolled back.")

        return segment_id

    def build_detection_segments(self, detections: list[DetectionRow], gap_tolerance_ms: int | None = None) -> list[
        Segment]:
        if gap_tolerance_ms is None:
            gap_tolerance_ms = self.gap_tolerance_ms
        segments: list[Segment] = []
        current: Optional[Segment] = None

        for row in detections:
            if current is not None:
                gap_ms = row.abs_start_ms - current.last_abs_end_ms
                same_species = row.species == current.species
                within_gap = gap_ms <= gap_tolerance_ms

                if same_species and within_gap:
                    current.last_abs_end_ms = row.abs_end_ms
                    current.last_datetime = row.end_datetime
                    current.end_sample = row.end_sample
                    current.last_chunk_id = row.chunk_id
                    current.chunk_ids.append(row.chunk_id)
                    current.members.append(row)
                    continue
                else:
                    current.segment_id = self._commit_segment(current)
                    segments.append(current)
                    current = None

            current = Segment(
                segment_type=SegmentType.DETECTION,
                species=row.species,
                first_abs_start_ms=row.abs_start_ms,
                last_abs_end_ms=row.abs_end_ms,
                first_chunk_id=row.chunk_id,
                last_chunk_id=row.chunk_id,
                chunk_ids=[row.chunk_id],
                members=[row],
                file_stem=row.file_stem,
                directory=row.directory,
                first_datetime=row.start_datetime,
                last_datetime=row.end_datetime,
                device=row.device,
                segment_id=current
            )

        if current is not None:
            current.segment_id = self._commit_segment(current)
            segments.append(current)

        return segments

    def _detection_clip_relpath(self, seg, sanitize_species):

        #species_slug = sanitize_species(seg.species)
        #return Path(f"{species_slug}_{seg.segment_id}.wav")
        return Path(f"segment_{seg.segment_id}.wav")

    def _cluster_clip_relpath(self, seg):

        return Path(f"segment_{seg.segment_id}.wav")


    def _insert_clip_file(self, segment_id, relpath):
        utilities = SQLServerUtilities(sp='sp_insert_segment_clip',
                                       sql_server_connection=self.sqlserver_connection,
                                       params_values=(segment_id, str(relpath)),
                                       params='@SegmentID=?, @FilePath=?', logger=self.logger)
        try:
            utilities.run_sql_params()
        except DatabaseIntegrityException:
            self.logger.error(f"ClipFiles row already exists for segment {segment_id}.")

    def _ensure_clip_file_row(self, segment_id, relpath):
        # covers a file that exists on disk but never got its DB row
        # (e.g. crash between write and commit on a prior run)
        self._insert_clip_file(segment_id, relpath)


    def carve_segment_clips(self, segments, output_path, make_relpath):
        cache = WavCache()
        written = []
        try:
            for seg in segments:
                relpath = make_relpath(seg)
                out_path = Path(output_path) / relpath

                if out_path.exists():
                    self._ensure_clip_file_row(seg.segment_id, relpath)
                    written.append((relpath.stem, str(out_path), seg.run_id, seg.cluster_id, seg.directory,
                                   (seg.species or '').replace(' ', '_')))
                    continue

                all_data = bytearray()
                wave_params = None

                for file_stem, file_members in groupby(seg.members, key=attrgetter("file_stem")):
                    file_members = list(file_members)
                    directory = file_members[0].directory

                    wf = cache.get(Path(self.audio_path / directory), file_stem)
                    params = cache.params(Path(self.audio_path / directory), file_stem)
                    wave_params = wave_params or params

                    data = self.extract_segment_bytes(wf, params, file_members[0].start_sample,
                                                      file_members[-1].end_sample)
                    if data:
                        all_data.extend(data)

                if not all_data:
                    print(f"Skipping empty segment: {seg.first_chunk_id}-{seg.last_chunk_id}")
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)

                with wave.open(str(out_path), "wb") as out_wf:
                    out_wf.setnchannels(wave_params.nchannels)
                    out_wf.setsampwidth(wave_params.sampwidth)
                    out_wf.setframerate(wave_params.framerate)
                    out_wf.writeframes(bytes(all_data))

                self._insert_clip_file(seg.segment_id, relpath)
                written.append((relpath.stem, str(out_path), seg.run_id, seg.cluster_id, seg.directory,
                                (seg.species or '').replace(' ', '_')))
            return written
        finally:
            cache.close_all()

    def _extract_link_fields(self, clip_record):
        """
        Pull run_id, cluster_id, and the source .wav path off a clip record
        returned by carve_segment_clips(). Works whether clip_record is a dict
        or an object with attributes -- adjust field names here if your
        actual clip records use different keys/attrs.
        """

        def _get(obj, name):
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        run_id = _get(clip_record, "run_id") or _get(clip_record, "RunID")
        cluster_id = _get(clip_record, "cluster_id") or _get(clip_record, "ClusterID")
        filepath = _get(clip_record, "filepath") or _get(clip_record, "path")

        if filepath is None:
            raise ValueError(f"Could not find a filepath on clip record: {clip_record!r}")
        if run_id is None or cluster_id is None:
            raise ValueError(f"Could not find run_id/cluster_id on clip record: {clip_record!r}")

        return run_id, cluster_id, Path(filepath)

    def build_cluster_link_tree(self, cluster_clip_records, links_root):

        links_root = Path(links_root)
        links_root.mkdir(parents=True, exist_ok=True)

        linked_count = 0
        skipped = []

        for record in cluster_clip_records:
            try:
                run_id, cluster_id, src_path = (record[2], record[3], Path(self.clips_path) / Path(record[0] + '.wav'))
            except ValueError as e:
                skipped.append((record, str(e)))
                self.logger.error(str(e))
                continue

            if not src_path.exists():
                msg = f"Source clip missing, skipping link: {src_path}"
                self.logger.error(msg)
                skipped.append((record, msg))
                continue
            # add suffix to cluster ID with total seconds length and num segments
            utilities = SQLServerUtilities(sp='sp_get_cluster_counts',
                                           sql_server_connection=self.sqlserver_connection,
                                           params_values=(run_id, cluster_id), params='@RunID=?, @ClusterID=?',
                                           logger=self.logger)
            data_stats = utilities.run_sql_return_params()[0]
            cluster_dir = links_root / f"run_{run_id}" / f"cluster_{cluster_id}_{data_stats[0]}min_{data_stats[1]}seg"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            link_path = cluster_dir / src_path.name

            # avoid crashing on a rebuild/re-run where the link already exists
            if link_path.exists():
                link_path.unlink()

            try:
                os.link(src_path, link_path)
                linked_count += 1
            except OSError as e:
                # Most common cause: links_root is on a different drive/volume
                # than src_path -- Windows hard links can't cross volumes.
                msg = f"Failed to hard link {src_path} -> {link_path}: {e}"
                self.logger.error(msg)
                skipped.append((record, msg))

        self.logger.info(
            f"Built cluster link tree at {links_root}: {linked_count} linked, {len(skipped)} skipped."
        )
        return linked_count, skipped

    def build_detection_link_tree(self, detection_clip_records, links_root):

        links_root = Path(links_root)
        links_root.mkdir(parents=True, exist_ok=True)

        linked_count = 0
        skipped = []

        for record in detection_clip_records:
            try:
                batch_dir, segment_id, src_path, species = (record[4], record[0],
                                                   Path(self.clips_path) / Path(record[0] + '.wav'), record[5])
            except ValueError as e:
                skipped.append((record, str(e)))
                self.logger.error(str(e))
                continue

            if not src_path.exists():
                msg = f"Source clip missing, skipping link: {src_path}"
                self.logger.error(msg)
                skipped.append((record, msg))
                continue

            batch_link_dir = links_root / str(batch_dir)
            batch_link_dir.mkdir(parents=True, exist_ok=True)

            link_path = batch_link_dir / f"{species.replace(' ', '-')}_{segment_id.split('_')[1]}.wav"

            # avoid crashing on a rebuild/re-run where the link already exists
            if link_path.exists():
                link_path.unlink()

            try:
                os.link(src_path, link_path)
                linked_count += 1
            except OSError as e:
                # Most common cause: links_root is on a different drive/volume
                # than src_path -- Windows hard links can't cross volumes.
                msg = f"Failed to hard link {src_path} -> {link_path}: {e}"
                self.logger.error(msg)
                skipped.append((record, msg))

        self.logger.info(
            f"Built detection link tree at {links_root}: {linked_count} linked, {len(skipped)} skipped."
        )
        return linked_count, skipped