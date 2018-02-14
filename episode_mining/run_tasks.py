from collections import defaultdict
import os

import numpy as np
import pandas as pd

from episode_mining.metrics.shao import ShaoMetric
from episode_mining.lib.utils import flatten


def get_datasource_metrics(datasource, disaggregator_class, config, output_dir):
    disaggregator = disaggregator_class(datasource, config)
    disaggregated_ts_map = disaggregator.get_disaggregated_timeseries()
    ground_truth_ts_map = datasource.get_ground_truth_disaggregated_timeseries()

    app_tp_map = defaultdict(int)
    app_fp_map = defaultdict(int)
    app_fn_map = defaultdict(int)

    for app in datasource.get_appliance_ids():
        if app not in ground_truth_ts_map.keys() or app not in disaggregated_ts_map.keys():
            continue

        ground_truth_ts = ground_truth_ts_map[app]
        disaggregated_ts = disaggregated_ts_map[app]

        if output_dir is not None:
            pd.Series(
                list(ground_truth_ts)
            ).to_csv(os.path.join(output_dir, 'gt_app_{}.csv'.format(app)))
            pd.Series(
                list(disaggregated_ts)
            ).to_csv(os.path.join(output_dir, 'disagg_app_{}.csv'.format(app)))

        metric = ShaoMetric(disaggregated_ts, ground_truth_ts, {
            'rho': 0.2,
            'theta': 30
        })
        app_tp_map[app] += metric.get_true_positives()
        app_fp_map[app] += metric.get_false_positives()
        app_fn_map[app] += metric.get_false_negatives()

    return app_tp_map, app_fp_map, app_fn_map


def run_disaggregator(disaggregator_class, config, datasources, output_dir=None):
    all_apps = flatten(
        [datasource.get_appliance_ids() for datasource in datasources]
    )
    app_id_to_label_map = dict(
        flatten([
            datasource.get_appliance_id_to_label_map().items()
            for datasource in datasources
        ])
    )

    results = [
        get_datasource_metrics(datasource, disaggregator_class, config, output_dir)
        for datasource in datasources
    ]

    app_tp_map = defaultdict(int)
    app_fp_map = defaultdict(int)
    app_fn_map = defaultdict(int)
    for r in results:
        r_tp_map, r_fp_map, r_fn_map = r
        for app, val in r_tp_map.items():
            app_tp_map[app] += val
        for app, val in r_fp_map.items():
            app_fp_map[app] += val
        for app, val in r_fn_map.items():
            app_fn_map[app] += val

    app_fm_map = {}
    app_p_map = {}
    app_r_map = {}

    def div_nan(a, b):
        if b == 0:
            b = np.nan
        return a/b

    for app in all_apps:
        if app not in (set(app_tp_map.keys()) & set(app_fp_map.keys()) & set(app_fn_map.keys())):
            app_p_map[app] = np.nan
            app_r_map[app] = np.nan
            app_fm_map[app] = np.nan
            continue
        app_p_map[app] = div_nan(app_tp_map[app], (app_tp_map[app] + app_fp_map[app]))
        app_r_map[app] = div_nan(app_tp_map[app], (app_tp_map[app] + app_fn_map[app]))
        app_fm_map[app] = div_nan(2, div_nan(1, app_p_map[app]) + div_nan(1, app_r_map[app]))

    out = pd.DataFrame(columns=('app_name', 'app_id', 'precision', 'recall', 'fmeasure'))

    for i, app in enumerate(all_apps):
        out.loc[i] = [
            app_id_to_label_map[app],
            app,
            app_p_map[app],
            app_r_map[app],
            app_fm_map[app]
        ]

    if output_dir is not None:
        out.to_csv(os.path.join(output_dir, 'metrics.csv'))

    return out