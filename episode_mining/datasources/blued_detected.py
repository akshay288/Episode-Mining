import pandas as pd

from episode_mining.datasources.blued_base import BLUEDBaseDatasource
from episode_mining.data_loaders import load_blued_cp_results
from episode_mining.lib.constants import SAME_CP_THRESH, Colls
from episode_mining.lib.utils import get_diffs


class BLUEDDetectedDatasource(BLUEDBaseDatasource):
    def __init__(self, phase, detected_results_path):
        super().__init__(phase)

        self.detected_results_path = detected_results_path
    
    def get_changepoints(self):
        cp_idxs, power_diffs, react_diffs = load_blued_cp_results(self.detected_results_path, self.phase)

        cp_idxs = pd.Series(cp_idxs)
        power_diffs = pd.Series(power_diffs)
        react_diffs = pd.Series(react_diffs)

        app_ids = []
        for i, cp_idx in enumerate(cp_idxs):
            app_id_idx = abs(self.blued_ground_truth[Colls.LOCATION] - cp_idx).idxmin()
            min_val = abs(self.blued_ground_truth[Colls.LOCATION] - cp_idx).min()

            if min_val < SAME_CP_THRESH:
                app_ids.append(self.blued_ground_truth[Colls.APP_ID][app_id_idx])
            else:
                app_ids.append(-1)

        return pd.DataFrame({
            Colls.LOCATION: cp_idxs,
            Colls.POWER_DIFFS: power_diffs,
            Colls.REACT_DIFFS: react_diffs,
            Colls.APP_ID: app_ids
        })