import glob
import ntpath
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from episode_mining.lib.constants import (
    Colls,
    BLUED_DATA_DIR,
    BLUED_GROUND_TRUTH_DATA_DIR
)


def load_blued_data():
    def load_blued_file(file_path):
        mat_data = loadmat(file_path)['data'][0,0]
        return pd.DataFrame.from_dict({
            'Pa': mat_data['Pa'][0],
            'Pb': mat_data['Pb'][0],
            'Qa': mat_data['Qa'][0],
            'Qb': mat_data['Qb'][0]
        })
    print('Loading BLUED Data...')
    files = glob.iglob(os.path.join(BLUED_DATA_DIR, '**/location*.mat'), recursive=True)
    data = []
    for file in tqdm(sorted(list(files))):
        data.append(load_blued_file(file)) 
    out = pd.concat(data, ignore_index=True)
    print('Done.')
    return out


def load_blued_ground_truth():
    print('Loading BLUED Ground Truth Data...')
    mat_data = loadmat(os.path.join(BLUED_GROUND_TRUTH_DATA_DIR, 'EVENTS_UNIQUE.mat'))['EVENTS_UNIQUE'][0,0]
    out = pd.DataFrame.from_dict({
        Colls.APP_ID: [e[0] for e in mat_data['appid']],
        Colls.APP_LABEL: [e[0][0] for e in mat_data['applabel']],
        Colls.PHASE: [e[0][0] for e in mat_data['phase']],
        Colls.LOCATION: [e[0] for e in mat_data['location']]
    })
    print('Done.')
    return out


def load_blued_cp_results(path, phase):
    print('Loading BLUED Changepoint Results...')
    mat_data = loadmat(path)
    cp_var = 'ChangePoint_Indexes_{}_global'.format(phase.upper())
    power_var = 'ChangePoint_Power_Diffs_{}_global'.format(phase.upper())
    react_var = 'ChangePoint_Reactive_Diffs_{}_global'.format(phase.upper())
    phase_cps = [cp[0] for cp in mat_data[cp_var]]
    phase_power_diffs = [cp[0] for cp in mat_data[power_var]]
    phase_react_diffs = [cp[0] for cp in mat_data[react_var]]
    print('Done.')
    return phase_cps, phase_power_diffs, phase_react_diffs


def load_redd_data(house_num, data_dir, has_timestamps=True):
    """
    Returns a dict with channel numbers as keys and a timeseries of power values
    as values
    """
    print('Loading REDD Data...')
    house_data_dir = os.path.join(data_dir, 'low_freq/house_{}'.format(house_num))
    files = glob.iglob(os.path.join(house_data_dir, 'channel*.dat'), recursive=True)

    raw_data = {}
    for file in tqdm(list(files)):
        channel_num = int(''.join([s for s in ntpath.basename(file) if s.isdigit()]))
        if has_timestamps:
            colls = names=('time', Colls.POWER)
        else:
            colls = (Colls.POWER,)
        channel_data = pd.read_csv(file, header=None, sep=' ', names=colls)
        raw_data[channel_num] = channel_data

    # Sum timeseries for appliances with the same label
    raw_labels = load_raw_redd_labels(house_num, data_dir)
    labels = load_redd_labels(house_num, data_dir)
    data = {}
    for i, row in labels.iterrows():
        app_id = row[Colls.APP_ID]
        app_label = row[Colls.APP_LABEL]
        channels = raw_labels[raw_labels[Colls.APP_LABEL] == app_label][Colls.APP_ID].tolist()
        data[app_id] = sum(raw_data[channel][Colls.POWER] for channel in channels)

    print('Done.')
    return data


def load_raw_redd_labels(house_num, data_dir):
    label_path = os.path.join(data_dir, 'low_freq/house_{}/labels.dat'.format(house_num))
    return pd.read_csv(label_path, header=None, names=(Colls.APP_ID, Colls.APP_LABEL), sep=' ')


def load_redd_labels(house_num, data_dir):
    raw_labels = load_raw_redd_labels(house_num, data_dir)
    labels = raw_labels[Colls.APP_LABEL].unique()
    labels = labels[labels != 'mains']
    return pd.DataFrame.from_dict({
        Colls.APP_ID: list(range(len(labels))),
        Colls.APP_LABEL: labels
    })


def load_redd_cp_results(path):
    print('Loading REDD Changepoint Results...')
    mat_data = loadmat(path)
    cp_data = mat_data['ChangePoint_Indexes'][0]
    main_cps = cp_data[0]
    main_cps = pd.Series([int(cp[0]) for cp in main_cps])
    print('Done Loading REDD Changepoint Results.')
    return main_cps