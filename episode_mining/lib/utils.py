from collections import defaultdict
import heapq
from itertools import combinations, cycle, islice

import numpy as np
import pandas as pd

from episode_mining.lib.constants import ZERO_THRESH
from episode_mining.lib.sparse_list import SparseList


def get_diffs(idxs, level_timeseries):
    """
    Returns a series with the difference in level_timeseries at each idx in idxs
    """
    prev_idxs = idxs.shift(1)
    prev_idxs[0] = 0

    next_idxs = idxs.shift(-1)
    next_idxs[len(next_idxs) - 1] = len(level_timeseries) - 1

    next_idxs = next_idxs.astype(int)
    prev_idxs = prev_idxs.astype(int)

    diffs = pd.Series(0.0, index=range(len(idxs)))
    for i, idx in enumerate(idxs):
        before = level_timeseries[prev_idxs[i]:idx].mean()
        after = level_timeseries[idx:next_idxs[i]].mean()
        diffs[i] = after - before
    
    return diffs

def diffs_to_levels(diffs, zero_thresh=None):
    if not zero_thresh:
        zero_thresh = ZERO_THRESH

    levels = []
    for diff in diffs:
        if len(levels) == 0:
            if diff > 0:
                levels.append(diff)
            else:
                levels.append(0)
        else:
            new_val = levels[-1] + diff
            if new_val < zero_thresh:
                new_val = 0
            levels.append(new_val)
    return levels

def levels_to_timeseries(levels, cp_idxs, timeseries_length):
    timeseries = SparseList(timeseries_length, fill_value=0.0)

    for i, level in enumerate(levels[:-1]):
        start = cp_idxs[i]
        finish = cp_idxs[i+1] + 1

        timeseries[start:finish] = level;

    timeseries[cp_idxs[-1]:len(timeseries)] = levels[-1]
    return timeseries

def zerofy_diffs(diffs):
    """
    makes the sum of the diff sequence zero by changing the last elemnt
    """
    diffs[-1] = -sum(diffs[:-1])
    return diffs

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

def get_event_combos(max_items, max_range):
    combos = []

    event_idxs = range(1, max_range + 1)
    for num_items in range(1, max_items):
        combos.append(combinations(event_idxs, num_items))

    return [(0,) + combo for combo in roundrobin(*combos)]

def level_seq_diff(level_seq_1, level_seq_2):
    if len(level_seq_1) != len(level_seq_2):
        return np.inf
    
    level_seq_1 = np.array(level_seq_1).astype(float)
    level_seq_2 = np.array(level_seq_2).astype(float)

    return np.average(
        abs(abs(level_seq_1 - level_seq_2) / np.average([level_seq_1, level_seq_2], axis=0))
    )

def flatten(ls):
    """
    only works one level deep
    https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    """
    return [item for sublist in ls for item in sublist]