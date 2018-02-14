from collections import defaultdict
import os
from itertools import combinations, dropwhile
import tempfile

import pandas as pd
import numpy as np
from heapq import merge
from tqdm import tqdm

from episode_mining.lib.constants import Colls, K_MEANS_DIR
from episode_mining.lib.utils import (
    get_event_combos,
    level_seq_diff,
    flatten,
    zerofy_diffs,
    diffs_to_levels,
    levels_to_timeseries
)

from episode_mining.disaggregators.episode_base import EpisodeBaseDisaggregator


class MotifDisaggregator(EpisodeBaseDisaggregator):

    def get_disaggregated_timeseries(self):
        changepoints = self.datasource.get_changepoints()

        gk_centers, cluster_dict = self.get_clusters(changepoints)

        num_centers = len(gk_centers)
        max_seq_length = self.config['max_seq_length']
        seq_lengths = range(2, max_seq_length + 1)
        combos = [combinations(range(num_centers), l) for l in seq_lengths]
        all_combos = merge(*combos)

        print('Finding event combos...')
        app_combos = []
        for combo in tqdm(list(all_combos)):
            power_diffs_valid = self.level_seq_valid(
                gk_centers[Colls.POWER_DIFFS][list(combo)]
            )
            react_diffs_valid = self.level_seq_valid(
                gk_centers[Colls.REACT_DIFFS][list(combo)]
            )

            if power_diffs_valid and react_diffs_valid:
                gk_centers.loc[combo, [Colls.POWER_DIFFS, Colls.REACT_DIFFS]] = np.nan
                app_combos.append(combo)
        print('Done.')

        print('Building episodes...')
        episodes = []
        for combo in tqdm(app_combos):
            combo_episodes = [sorted(cluster_dict[cluster]) for cluster in combo]
            while(all(len(e) > 0 for e in combo_episodes)):
                episode = []
                episode.append(combo_episodes[0].pop(0))
                for i in range(len(combo_episodes))[1:]:
                    combo_episodes[i] = list(
                        dropwhile(
                            lambda x: x <= episode[i-1],
                            combo_episodes[i]
                        )
                    )
                    if len(combo_episodes[i]) == 0:
                        episode = None
                        break
                    episode.append(combo_episodes[i].pop(0))
                if episode is not None:
                    episodes.append(episode)
        paired_cps = flatten(episodes)
        assert len(paired_cps) == len(set(paired_cps)) # can't pair a cp twice
        print('Done. (paired {}/{} changepoints)'.format(len(paired_cps), len(changepoints)))

        return self.episodes_to_ts(episodes)

    def get_clusters(self, changepoints):
        if Colls.REACT_DIFFS not in changepoints.columns.values:
            changepoints[Colls.REACT_DIFFS] = 0

        _, k_means_input  = tempfile.mkstemp()
        _, k_means_out  = tempfile.mkstemp()
        _, k_means_centers  = tempfile.mkstemp()

        out = changepoints[[Colls.POWER_DIFFS, Colls.REACT_DIFFS]]
        out = out.round(decimals=4)
        out.to_csv(k_means_input, header=False, index=False)

        os.system(
            ' '.join([
                os.path.join(K_MEANS_DIR, 'gka'),
                k_means_input,
                k_means_out,
                k_means_centers,
                str(150),
                str(len(changepoints.index))
            ])
        )

        gk_output = pd.read_csv(
            k_means_out,
            sep=' ',
            header=None,
            names=(Colls.POWER_DIFFS, Colls.REACT_DIFFS, Colls.CLUSTER_NUM)
        )
        gk_centers = pd.read_csv(
            k_means_centers,
            sep=' ',
            header=None,
            names=(Colls.POWER_DIFFS, Colls.REACT_DIFFS)
        )

        cluster_dict = defaultdict(list)
        for i, cluster_num in enumerate(gk_output[Colls.CLUSTER_NUM].tolist()):
            cluster_dict[cluster_num].append(i)

        return gk_centers, cluster_dict