import numpy as np
import pandas as pd

from tqdm import tqdm

from episode_mining.data_loaders import (
    load_redd_labels,
    load_redd_data,
    load_redd_cp_results
)
from episode_mining.datasources.base import BaseDatasource
from episode_mining.lib.constants import REDD_DATA_DIR, Colls
from episode_mining.lib.sparse_list import SparseList
from episode_mining.lib.utils import get_diffs


class REDDDatasource(BaseDatasource):
    def __init__(self, house_num, detected_results_path):
        super().__init__()

        self.house_num = house_num
        self.detected_results_path = detected_results_path

        self.redd_labels = load_redd_labels(house_num, REDD_DATA_DIR)
        self.redd_data = load_redd_data(house_num, REDD_DATA_DIR)

    def get_aggregate_timeseries(self):
        app_ids = self.get_appliance_ids()
        return sum(map(lambda app_id: self.redd_data[app_id], app_ids))

    def get_appliance_ids(self):
        return self.redd_labels[Colls.APP_ID].tolist()

    def get_appliance_id_to_label_map(self):
        out = self.redd_labels
        return dict(zip(out[Colls.APP_ID].tolist(), out[Colls.APP_LABEL].tolist()))

    def get_ground_truth_disaggregated_timeseries(self):
        app_ids = self.get_appliance_ids()
        return {k: SparseList.from_list(v.tolist()) for (k, v) in self.redd_data.items() if k in app_ids}

    def find_cp_app(self, cp, cp_results, app_ids):
       return app_ids[
            np.argmin([abs(cp_results[app - 1] - cp).min() for app in app_ids])
        ]

    def _power_diffs_per_app(self, cp_idxs):
        app_ids = self.get_appliance_ids()
        return {app: get_diffs(cp_idxs, self.redd_data[app]) for app in app_ids}

    def _find_cp_apps(self, cp_idxs):
        power_diffs_per_app = self._power_diffs_per_app(cp_idxs)
        apps = self.get_appliance_ids()

        cp_apps = []
        for i in range(len(cp_idxs)):
            cp_apps.append(max(apps, key= lambda app: abs(power_diffs_per_app[app][i])))

        return cp_apps

    def get_changepoints(self):
        app_ids = self.get_appliance_ids()
        cp_idxs = load_redd_cp_results(self.detected_results_path)
        power_diffs = get_diffs(cp_idxs, self.get_aggregate_timeseries())
        cp_apps = self._find_cp_apps(cp_idxs)
        return pd.DataFrame({
            Colls.APP_ID: cp_apps,
            Colls.LOCATION: cp_idxs,
            Colls.POWER_DIFFS: power_diffs,
        })