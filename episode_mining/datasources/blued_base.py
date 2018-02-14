from tqdm import tqdm

from episode_mining.data_loaders import load_blued_data, load_blued_ground_truth
from episode_mining.datasources.base import BaseDatasource
from episode_mining.lib.constants import Colls
from episode_mining.lib.utils import get_diffs, diffs_to_levels, levels_to_timeseries


class BLUEDBaseDatasource(BaseDatasource):
    def __init__(self, phase):
        super().__init__()
        
        self.phase = phase
        self.blued_data_power_coll = 'P{}'.format(self.phase.lower())
        self.blued_data_react_coll = 'Q{}'.format(self.phase.lower())

        self.blued_data = load_blued_data()

        self.blued_ground_truth = load_blued_ground_truth()
        self.blued_ground_truth = self.blued_ground_truth.loc[
            self.blued_ground_truth[Colls.PHASE] == self.phase.upper()
        ]
        self.blued_ground_truth.reset_index(inplace=True, drop=True)
        self.blued_ground_truth[Colls.POWER_DIFFS] = get_diffs(
            self.blued_ground_truth[Colls.LOCATION],
            self.blued_data[self.blued_data_power_coll]
        )
        self.blued_ground_truth[Colls.REACT_DIFFS] = get_diffs(
            self.blued_ground_truth[Colls.LOCATION],
            self.blued_data[self.blued_data_react_coll]
        )

    def get_aggregate_timeseries(self):
        """
        returns a dataframe with power and reactive power
        (labeled power and react)
        """
        out = self.blued_data.copy()

        out[Colls.POWER] = out[self.blued_data_power_coll] 
        out[Colls.REACT] = out[self.blued_data_react_coll] 

        return out[[Colls.POWER, Colls.REACT]]

    def get_appliance_ids(self):
        ids = self.blued_ground_truth['appid'].unique().tolist()
        return [i for i in ids if i < 1000]

    def get_appliance_id_to_label_map(self):
        out = self.blued_ground_truth[[Colls.APP_ID, Colls.APP_LABEL]]
        out = out.drop_duplicates()
        out = out[out[Colls.APP_ID].isin(self.get_appliance_ids())]
        return dict(zip(out[Colls.APP_ID].tolist(), out[Colls.APP_LABEL].tolist()))

    def get_ground_truth_disaggregated_timeseries(self):
        print('Getting ground truth disaggregated timeseries...')
        app_ids =  self.get_appliance_ids()
        out = {}
        for app_id in tqdm(app_ids):
            app_gt = self.blued_ground_truth[self.blued_ground_truth[Colls.APP_ID] == app_id]
            power_diffs = app_gt[Colls.POWER_DIFFS]
            cp_idxs = app_gt[Colls.LOCATION].tolist()

            levels = diffs_to_levels(power_diffs)
            ts = levels_to_timeseries(levels, cp_idxs, len(self.blued_data))

            out[app_id] = ts
        print('Done.')
        return out