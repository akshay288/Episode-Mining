from episode_mining.data_loaders import (
    load_redd_labels,
    load_redd_data
)
from episode_mining.lib.constants import ECOLABS_DATA_DIR
from episode_mining.datasources.redd import REDDDatasource


class EcolabDatasource(REDDDatasource):
    def __init__(self, house_num, detected_results_path):
        self.house_num = house_num
        self.detected_results_path = detected_results_path

        self.redd_labels = load_redd_labels(house_num, ECOLABS_DATA_DIR)
        self.redd_data = load_redd_data(house_num, ECOLABS_DATA_DIR, has_timestamps=False)
