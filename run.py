from birdnet_detect_and_cluster import BirdNetParser
from configs import config
from pathlib import Path
from ml_parameters.ml_parameters import UMAPParams, HDBSCANClusters
from connection import SQLServerConnection
import os, datetime, logging
from mu_utilities.utilities import SQLServerUtilities

dir_path = os.getcwd()
APP_NAME = dir_path[dir_path.rindex('/')+1:]
LOG_FILE_PATH = config.LOG_FILE_PATH
LOG_MESSAGE = '%(asctime)s -%(process)d - %(levelname)s - %(message)s'
LOG_TIME = '%d-%b-%y %H:%M:%S'
EXTERNAL_DRIVE = config.EXTERNAL_DRIVE
AUDIO_PATH = config.ROOT_DATA_PATH / Path("Input")
OUTPUT_PATH = config.ROOT_DATA_PATH / Path("Output")
SPECIES_LIST = config.ROOT_DATA_PATH / Path("species") / config.SPECIES_LIST
MIN_CONFIDENCE = config.MIN_CONFIDENCE
OVERLAP = config.OVERLAP
GAP_MS = config.GAP_MS
MIN_CLUSTER_SIZE = config.MIN_CLUSTER_SIZE
MIN_SAMPLES = config.MIN_SAMPLES
CLUSTER_METRIC = config.CLUSTER_METRIC
CLUSTER_SELECTION_EPSILON = config.CLUSTER_SELECTION_EPSILON
N_NEIGHBORS = config.N_NEIGHBORS
MIN_DISTANCE = config.MIN_DISTANCE
N_COMPONENTS = config.N_COMPONENTS
NMAP_METRIC = config.NMAP_METRIC
RANDOM_STATE = config.RANDOM_STATE
N_NEIGHBORS_VIZ = config.N_NEIGHBORS_VIZ
MIN_DISTANCE_VIZ = config.MIN_DISTANCE_VIZ
N_COMPONENTS_VIZ = config.N_COMPONENTS_VIZ
NMAP_METRIC_VIZ = config.NMAP_METRIC_VIZ
RANDOM_STATE_VIZ = config.RANDOM_STATE_VIZ
ANALYSIS_RUN_TEXT = config.ANALYSIS_RUN_TEXT
ANALYZE_FILE_GROUP = config.ANALYZE_FILE_GROUP
SQLSERVER_NAME = config.SQLSERVER_NAME
SQLSERVER_DATABASE = config.SQLSERVER_DATABASE
SQLSERVER_USERNAME = config.SQLSERVER_USERNAME
SQLSERVER_TOKEN = config.SQLSERVER_TOKEN
SQLSERVER_KEY = config.SQLSERVER_KEY


def initialize_logger(directory):
    full_path = LOG_FILE_PATH + '/' + datetime.datetime.now().strftime("%Y-%m-%d") + '.log'
    global_format = logging.Formatter(LOG_MESSAGE, LOG_TIME)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_logger = logging.FileHandler(full_path)
    file_logger.setLevel(logging.INFO)
    file_logger.setFormatter(global_format)
    logger.addHandler(file_logger)
    return logger

def initialize_umap():
    umap_params = UMAPParams(n_neighbors=N_NEIGHBORS, min_distance=MIN_DISTANCE, random_state=RANDOM_STATE,
                             n_components=N_COMPONENTS, metric=NMAP_METRIC)
    return umap_params


def initialize_umap_viz():
    umap_params_viz = UMAPParams(n_neighbors=N_NEIGHBORS_VIZ, min_distance=MIN_DISTANCE_VIZ,
                                 random_state=RANDOM_STATE_VIZ, n_components=N_COMPONENTS_VIZ, metric=NMAP_METRIC_VIZ)

    return umap_params_viz


def initialize_hdbscan():
    hdbscan_params = HDBSCANClusters(min_samples=MIN_SAMPLES, min_cluster_size=MIN_CLUSTER_SIZE,
                                     cluster_metric=CLUSTER_METRIC, cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON)
    return hdbscan_params


def initialize_sqlserver():
    sqlserver_connection = SQLServerConnection(name=SQLSERVER_NAME, database=SQLSERVER_DATABASE,
                                               username=SQLSERVER_USERNAME, key=SQLSERVER_KEY, token=SQLSERVER_TOKEN)
    return sqlserver_connection


parse = BirdNetParser(logger=initialize_logger('chipbot_parser'), audio_path=AUDIO_PATH, output_path=OUTPUT_PATH,
                      min_confidence=MIN_CONFIDENCE, species_list=SPECIES_LIST,
                      external_drive=EXTERNAL_DRIVE,
                      gap_ms=GAP_MS, umap=initialize_umap(), umap_viz=initialize_umap_viz(),
                      hdbscan_clusters=initialize_hdbscan(), analysis_run_text=ANALYSIS_RUN_TEXT,
                      analyze_file_group=ANALYZE_FILE_GROUP, overlap=OVERLAP,
                      sqlserver_connection=initialize_sqlserver())

parse.run_pipeline()
