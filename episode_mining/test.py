#!/usr/bin/env python

import os

from episode_mining.datasources.blued_detected import BLUEDDetectedDatasource
from episode_mining.datasources.blued_ground_truth import BLUEDGroundTruthDatasource
from episode_mining.datasources.redd import REDDDatasource
from episode_mining.datasources.ecolab import EcolabDatasource
from episode_mining.disaggregators.iterative import IterativeDisaggregator
from episode_mining.disaggregators.motif import MotifDisaggregator

from episode_mining.run_tasks import run_disaggregator

# ds_a = BLUEDDetectedDatasource('a', '/Users/akshay/Documents/Projects/episode_mining/DATA/BLUED-Results/60Hz.mat')
# ds_b = BLUEDDetectedDatasource('b', '/Users/akshay/Documents/Projects/episode_mining/DATA/BLUED-Results/60Hz.mat')
# ds_a = BLUEDGroundTruthDatasource('a')
# ds_b = BLUEDGroundTruthDatasource('b')
ds = REDDDatasource(1, '/Users/akshay/Documents/Projects/episode_mining/DATA/REDD-Results/Results_CP_house_1.mat')
# ds = EcolabDatasource(2, '/Users/akshay/Documents/Projects/episode_mining/DATA/ecolabs_results/Results_CP_house_2.mat')

os.makedirs('test')

metrics = run_disaggregator(
    IterativeDisaggregator,
    {
    'min_range': 5,
    'max_range': 20,
    'step_size': 5,
    'max_seq_length': 3,
    'zero_thresh': 65,
    'max_cluster_diff': 0.000001
    },
    [ds],# , ds_b]
    output_dir='test'
)

print(metrics)