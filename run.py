from birdnet_detect_and_cluster import BirdNetParser
from configs import config

EXTERNAL_DRIVE = config.EXTERNAL_DRIVE
AUDIO_PATH = config.ROOT_DATA_PATH + "\\input"
OUTPUT_PATH = config.ROOT_DATA_PATH + "\\output"
SPECIES_LIST = config.ROOT_DATA_PATH + "\\species\\" + config.SPECIES_LIST
MIN_CONFIDENCE_OUTPUT = config.MIN_CONFIDENCE_OUTPUT
MIN_CONFIDENCE_INPUT = config.MIN_CONFIDENCE_INPUT
GAP_MS = config.GAP_MS

parse = BirdNetParser(logger='', audio_path=AUDIO_PATH, output_path=OUTPUT_PATH,
                      min_confidence_input=MIN_CONFIDENCE_INPUT, species_list=SPECIES_LIST,
                      external_drive=EXTERNAL_DRIVE, min_confidence_output=MIN_CONFIDENCE_OUTPUT,
                      gap_ms=GAP_MS)

parse.run_pipeline()
