from birdnet_detect_and_cluster import BirdNetParser
from configs import config


EXTERNAL_DRIVE = config.EXTERNAL_DRIVE
AUDIO_PATH = config.ROOT_DATA_PATH + "\\input"
OUTPUT_PATH = config.ROOT_DATA_PATH + "\\output"
SPECIES_LIST = config.ROOT_DATA_PATH + "\\species\\" + config.SPECIES_LIST
MIN_CONFIDENCE = config.MIN_CONFIDENCE

parse = BirdNetParser(logger='', audio_path=AUDIO_PATH, output_path=OUTPUT_PATH,
                      min_confidence=MIN_CONFIDENCE, species_list=SPECIES_LIST, external_drive=EXTERNAL_DRIVE)

parse.run_pipeline()
