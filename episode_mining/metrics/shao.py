import numpy as np

from episode_mining.lib.utils import flatten
from episode_mining.metrics.base import BaseMetric


class ShaoMetric(BaseMetric):
    def __init__(self, disaggregated_ts, ground_truth_ts, config):
        super().__init__(disaggregated_ts, ground_truth_ts, config)

        assert len(self.disaggregated_ts) == len(self.ground_truth_ts)
        
        ts_len = len(self.disaggregated_ts)

        disaggregated_intervals = self.disaggregated_ts.interval_val_map.keys()
        ground_truth_intervals = self.ground_truth_ts.interval_val_map.keys()
        disaggregated_cps = set(flatten([[interval[0], interval[1]] for interval in disaggregated_intervals]))
        ground_truth_cps = set(flatten([[interval[0], interval[1]] for interval in ground_truth_intervals]))

        all_cps = {*disaggregated_cps, *ground_truth_cps}

        # rho is the wiggle room for correctness
        rho = self.config['rho']
        # theta is the minimum value considered to be on
        theta = self.config['theta']

        tp = 0
        fn = 0
        fp = 0

        timesteps = [0] + sorted(all_cps)[:-1]
        diffs = list(np.array(timesteps[1:]) - np.array(timesteps[:-1])) + [ts_len - timesteps[-1]]
        timesteps = [int(t) for t in timesteps]
        disaggregated_levels = [disaggregated_ts[t] for t in timesteps]
        ground_truth_levels = [ground_truth_ts[t] for t in timesteps]

        for disaggregated_level, ground_truth_level, diff in list(zip(disaggregated_levels, ground_truth_levels, diffs)):
            if disaggregated_level >= theta and ground_truth_level >= theta:
                if (ground_truth_level * (1-rho)) <= disaggregated_level and disaggregated_level <= (ground_truth_level * (1+rho)):
                    tp += disaggregated_level * diff
                elif disaggregated_level < (ground_truth_level * (1-rho)):
                    tp += disaggregated_level * diff
                    fn += (ground_truth_level - disaggregated_level) * diff
                elif disaggregated_level > (ground_truth_level * (1+rho)):
                    tp += disaggregated_level * diff
                    fp += (disaggregated_level - ground_truth_level) * diff
            elif (ground_truth_level >= theta) and (disaggregated_level < theta):
                fn += ground_truth_level * diff
            elif (disaggregated_level >= theta) and (ground_truth_level < theta):
                fp += disaggregated_level * diff

        self.tp = tp
        self.fp = fp
        self.fn = fn


    def get_true_positives(self):
        return self.tp
    
    def get_false_positives(self):
        return self.fp
    
    def get_false_negatives(self):
        return self.fn

