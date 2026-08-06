from birdnet_detect_and_cluster import BirdNetParser
from configs import config
from ml_parameters.ml_parameters import UMAPParams, HDBSCANClusters

EXTERNAL_DRIVE = config.EXTERNAL_DRIVE
AUDIO_PATH = config.ROOT_DATA_PATH + "\\input"
OUTPUT_PATH = config.ROOT_DATA_PATH + "\\output"
SPECIES_LIST = config.ROOT_DATA_PATH + "\\species\\" + config.SPECIES_LIST
MIN_CONFIDENCE_OUTPUT = config.MIN_CONFIDENCE_OUTPUT
MIN_CONFIDENCE_INPUT = config.MIN_CONFIDENCE_INPUT
GAP_MS = config.GAP_MS
MIN_CLUSTER_SIZE = config.MIN_CLUSTER_SIZE
MIN_SAMPLES = config.MIN_SAMPLES
CLUSTER_METRIC = config.CLUSTER_METRIC
N_NEIGHBORS = config.N_NEIGHBORS
MIN_DISTANCE = config.MIN_DISTANCE
N_COMPONENTS = config.N_COMPONENTS
NMAP_METRIC = config.NMAP_METRIC
RANDOM_STATE = config.RANDOM_STATE
N_NEIGHBORS_SEC = config.N_NEIGHBORS_SEC
MIN_DISTANCE_SEC = config.MIN_DISTANCE_SEC
N_COMPONENTS_SEC = config.N_COMPONENTS_SEC
NMAP_METRIC_SEC = config.NMAP_METRIC_SEC
RANDOM_STATE_SEC = config.RANDOM_STATE_SEC
ANALYSIS_RUN_TEXT = config.ANALYSIS_RUN_TEXT
ANALYZE_FILE_GROUP = config.ANALYZE_FILE_GROUP


def initialize_umap():
    umap_params = UMAPParams(n_neighbors=N_NEIGHBORS, min_distance=MIN_DISTANCE, random_state=RANDOM_STATE,
                             n_components=N_COMPONENTS, metric=NMAP_METRIC)
    return umap_params


def initialize_umap_sec():
    umap_params_sec = UMAPParams(n_neighbors=N_NEIGHBORS_SEC, min_distance=MIN_DISTANCE_SEC,
                                 random_state=RANDOM_STATE_SEC, n_components=N_COMPONENTS_SEC, metric=NMAP_METRIC_SEC)

    return umap_params_sec


def initialize_hdbscan():
    hdbscan_params = HDBSCANClusters(min_samples=MIN_SAMPLES, min_cluster_size=MIN_CLUSTER_SIZE,
                                     cluster_metric=CLUSTER_METRIC)
    return hdbscan_params


parse = BirdNetParser(logger='', audio_path=AUDIO_PATH, output_path=OUTPUT_PATH,
                      min_confidence_input=MIN_CONFIDENCE_INPUT, species_list=SPECIES_LIST,
                      external_drive=EXTERNAL_DRIVE, min_confidence_output=MIN_CONFIDENCE_OUTPUT,
                      gap_ms=GAP_MS, umap=initialize_umap(), umap_second=initialize_umap_sec(),
                      hdbscan_clusters=initialize_hdbscan(), analysis_run_text=ANALYSIS_RUN_TEXT,
                      analyze_file_group=ANALYZE_FILE_GROUP)

parse.run_pipeline()
