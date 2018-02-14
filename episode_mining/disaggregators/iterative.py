from collections import defaultdict

import numpy as np
from tqdm import tqdm
import pandas as pd

from episode_mining.lib.constants import Colls
from episode_mining.lib.utils import (
    get_event_combos,
    level_seq_diff,
    flatten,
    zerofy_diffs,
    diffs_to_levels,
    levels_to_timeseries
)

from episode_mining.disaggregators.episode_base import EpisodeBaseDisaggregator


class IterativeDisaggregator(EpisodeBaseDisaggregator):

    def get_disaggregated_timeseries(self):
        changepoints = self.datasource.get_changepoints()

        # get config values
        min_range = self.config['min_range']
        max_range = self.config['max_range']
        step_size = self.config['step_size']

        # find episodes
        episodes = []
        search_depths = list(range(min_range, max_range, step_size)) + [max_range]

        print('Finding episodes...')
        diffs = changepoints[Colls.POWER_DIFFS].copy()
        for cp_idx in tqdm(range(len(changepoints))):
            episode, diffs = self.find_idx_episode(cp_idx, diffs, search_depths)
            if episode:
                episodes.append(episode)
        paired_cps = flatten(episodes)
        assert len(paired_cps) == len(set(paired_cps)) # can't pair a cp twice
        print('Done Finding episodes. (paired {}/{} changepoints)'.format(len(paired_cps), len(changepoints)))

        return self.episodes_to_ts(episodes)

    def find_idx_episode(self, cp_idx, levels, search_depths):
        max_episodes = self.config['max_seq_length']
        num_changepoints = len(levels)
        for depth in search_depths:
            if (cp_idx + depth) > num_changepoints:
                depth = num_changepoints - cp_idx

            possible_combos = get_event_combos(max_episodes, depth)

            for combo in possible_combos:
                episode = [cp_idx + offset for offset in combo]
                episode_levels = levels[episode]

                if self.level_seq_valid(episode_levels):
                    levels.loc[episode] = np.nan
                    return episode, levels

        return None, levels