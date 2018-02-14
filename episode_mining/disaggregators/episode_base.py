from collections import defaultdict

import numpy as np
from tqdm import tqdm

from episode_mining.lib.constants import Colls
from episode_mining.lib.utils import (
    get_event_combos,
    level_seq_diff,
    flatten,
    zerofy_diffs,
    diffs_to_levels,
    levels_to_timeseries
)

from episode_mining.disaggregators.base import BaseDisaggregator


class EpisodeBaseDisaggregator(BaseDisaggregator):
    
    def level_seq_valid(self, levels):
        zero_thresh = self.config['zero_thresh']

        if any(np.isnan(levels)):
            return False

        for prefix_length in range(1, len(levels) + 1):
            if sum(levels[:prefix_length]) < 0:
                return False

        if abs(sum(levels)) > zero_thresh:
            return False

        # TODO Either make zero_thresh a percent or get rid of low power cps

        return True

    def episodes_to_ts(self, episodes):
        changepoints = self.datasource.get_changepoints()

        # cluster episodes
        print('Clustering episodes...')
        clusters_with_diffs = []
        clusters = []
        episode_diffs = [
            changepoints[Colls.POWER_DIFFS][episode].tolist()
            for episode in episodes
        ]
        for episode_idx, episode_diffs in enumerate(episode_diffs):
            found = False
            for cluster_idx, cluster in enumerate(clusters):
                cluster_diffs = clusters_with_diffs[cluster_idx]
                cluster_diffs = np.average(cluster_diffs, axis=0)
                if level_seq_diff(cluster_diffs, episode_diffs) < self.config['max_cluster_diff']:
                    clusters[cluster_idx].append(episode_idx)
                    clusters_with_diffs[cluster_idx].append(episode_diffs)
                    found = True
                    break
            if not found:
                clusters.append([episode_idx])
                clusters_with_diffs.append([episode_diffs])
        print('Done Clustering episodes. (found {} clusters)'.format(len(clusters)))

        # find cluster appliances
        # Using ground truth - no better way to do this
        cluster_apps = []
        for cluster in clusters:
            cluster_cp_idxs = flatten([episodes[ep_idx] for ep_idx in cluster])
            cluster_apps.append(min(changepoints[Colls.APP_ID][cluster_cp_idxs].mode()))

        # find appliance episodes
        app_episode_map = defaultdict(list)
        for cluster_app, cluster_episodes in zip(cluster_apps, clusters):
            app_episode_map[cluster_app].extend(cluster_episodes)

        # get all the cps and diffs for an app
        app_cp_map = defaultdict(list)
        app_diff_map = defaultdict(list)
        for app, app_episodes in app_episode_map.items():
            app_episodes = [episodes[idx] for idx in app_episodes]
            cp_idxs = flatten(app_episodes)
            app_diffs = []
            for episode in app_episodes:
                # app_diffs.extend(zerofy_diffs(changepoints[Colls.POWER_DIFFS][episode].tolist()))
                app_diffs.extend(changepoints[Colls.POWER_DIFFS][episode].tolist())

            app_cp_map[app].extend(changepoints[Colls.LOCATION][cp_idxs].tolist())
            app_diff_map[app].extend(app_diffs)

        # get the app dissagregated timeseries
        print('Building Timeseries...')
        ts_len = len(self.datasource.get_aggregate_timeseries())
        app_ts_map = {}
        for app, app_diffs in tqdm(app_diff_map.items()):
            app_cps = app_cp_map[app]
            
            # sort diffs based on the changepoint index
            app_diffs = [diff[1] for diff in
                         sorted(enumerate(app_diffs), key=lambda x: app_cps[x[0]])]
            app_cps = sorted(app_cps)
            levels = diffs_to_levels(app_diffs, zero_thresh=0.0000001)

            app_ts_map[app] = levels_to_timeseries(
                levels,
                app_cps,
                ts_len
            )
        print('Done.')

        return app_ts_map          