#!/usr/bin/env python

import argparse
import os
import subprocess
import shutil

import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', required=True)
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-d', '--disclude_labels', nargs='+', default=[])

args = parser.parse_args()

output_path = os.path.join(args.output_dir, 'low_freq')
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

houses = sorted(
    [name for name in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, name))]
)

for house in tqdm(houses):
    house_data_dir = os.path.join(args.input_dir, house, 'data')
    house_label_path = os.path.join(args.input_dir, house, 'doc.txt')

    labels = pd.read_csv(
        house_label_path, header=None, dtype=str, names=('id', 'label'), sep=' '
    )
    labels = labels[~labels.label.isin(args.disclude_labels)]
    labels['real_id'] = [3 + rID for rID in range(len(labels))]

    days = {}
    for app_id in labels['id']:
        days[app_id] = os.listdir(os.path.join(house_data_dir, app_id))

    valid_days = sorted(
        list(
            set.intersection(*[set(ls) for ls in days.values()])
        )
    )
    
    output_house_path = os.path.join(output_path, house)
    os.mkdir(output_house_path)

    outputs = []

    for app_id, real_id in zip(labels['id'], labels['real_id']):
        output_file_path = os.path.join(
            output_house_path, 'channel_{}.dat'.format(real_id)
        )
        output_file = open(output_file_path, 'w')
        outputs.append(output_file_path)
        input_files = [
            os.path.join(house_data_dir, app_id, day) for day in valid_days
        ]
        for f in input_files:
            subprocess.Popen(['cat', f], stdout=output_file)

    synth = sum([pd.read_csv(output, header=None) for output in outputs])

    synth.to_csv(
        os.path.join(output_house_path, 'channel_1.dat'),
        index=False, header=False, sep=' '
    )
    synth.to_csv(
        os.path.join(output_house_path, 'channel_2.dat'),
        index=False, header=False, sep=' '
    )

    labels.loc[-2] = [-2, 'mains', 1]
    labels.loc[-1] = [-1, 'mains', 2]
    labels = labels.sort_values('real_id')
    labels[['real_id', 'label']].to_csv(
        os.path.join(output_house_path, 'labels.dat'),
        index=False, header=False, sep=' '
    )