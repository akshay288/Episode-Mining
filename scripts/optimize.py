#!/usr/bin/env python

import glob
from itertools import product
import json
import multiprocessing as mp
import os

from episode_mining.datasources.blued_detected import BLUEDDetectedDatasource
from episode_mining.datasources.blued_ground_truth import BLUEDGroundTruthDatasource
from episode_mining.datasources.redd import REDDDatasource
from episode_mining.datasources.ecolab import EcolabDatasource
from episode_mining.disaggregators.iterative import IterativeDisaggregator
from episode_mining.disaggregators.motif import MotifDisaggregator

from episode_mining.run_tasks import run_disaggregator


# The different options to change
iterative_config = {
    'min_range': 5,
    'max_range': 20,
    'step_size': 5,
    'max_seq_length': None,
    'zero_thresh': None,
    'max_cluster_diff': 0.000001
}
motif_config = {
    'max_seq_length': None,
    'zero_thresh': None,
    'max_cluster_diff': 0.000001
}
seq_lengths = range(2, 3)
zero_threshs = range(5, 15, 5)
datasources = [
    REDDDatasource(1, '/Users/akshay/Documents/Projects/episode_mining/DATA/REDD-Results/Results_CP_house_1.mat')
]
disaggregator = IterativeDisaggregator
config = iterative_config
output_dir = 'test_redd_iterative_house_1'

os.mkdir(output_dir)
options = list(product(seq_lengths, zero_threshs))

def get_option_path(seq_length, zero_thresh):
    return '{}/seq_length_{}_zero_thresh_{}'.format(
        output_dir, seq_length, zero_thresh
    )

def disaggregation_helper(option):
    seq_length = option[0]
    zero_thresh = option[1]
    option_output_dir = get_option_path(seq_length, zero_thresh)

    os.mkdir(option_output_dir)
    config['max_seq_length'] = seq_length
    config['zero_thresh'] = zero_thresh

    metrics = run_disaggregator(
        disaggregator,
        config,
        datasources,
        output_dir=option_output_dir
    )

    with open(os.path.join(option_output_dir, 'config.json'), 'w') as f:
        f.write(
            json.dumps(config, indent=4, sort_keys=True)
        )

    return metrics

pool = mp.Pool(processes=8)
metrics = pool.map(disaggregation_helper, options)

max_metrics_avg = -1
max_metrics = None
max_option = None
for i, m in enumerate(metrics):
    metric_avg = m['fmeasure'].mean()
    if metric_avg > max_metrics_avg:
        max_metrics_avg = metric_avg
        max_metrics = m
        max_option = options[i]

with open(os.path.join(output_dir, 'max_results.txt'), 'w') as f:
    f.write('Max Seq Length: {}\n'.format(max_option[0]))
    f.write('Max Zero Thresh: {}\n'.format(max_option[1]))
    f.write(str(max_metrics))
    f.write('\n')
    f.write('\n')
    f.write('Average:\n')
    f.write(str(max_metrics[['precision', 'recall', 'fmeasure']].mean(axis=0)))

# clean up unecessacery timeseries
for option in options:
    if option != max_option:
        for f in glob.glob(os.path.join(get_option_path(*option), '*.csv')):
            if 'metrics.csv' not in f:
                os.remove(f)