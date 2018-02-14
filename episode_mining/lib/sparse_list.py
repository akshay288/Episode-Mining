from bisect import bisect_left
from operator import add, sub, mul

import numpy as np
from sortedcontainers import SortedDict


class SparseList(object):
    """
    SparseList implements most functionalities of a 1d matrix efficiently when
    the size of the list is large but the data is sparse.
    TODO rewrite with an interval class
    """

    def __init__(self, list_size, fill_value=np.nan):
        self.interval_val_map = SortedDict()
        self.list_size = list_size
        self.fill_value = fill_value
        self.curr_pos = 0

    @classmethod
    def from_list(cls, ls, fill_value=0.0):
        sparse_ls = cls(len(ls), fill_value=fill_value)

        nparr = np.array(ls)
        nparr = np.around(nparr, decimals=-1)
        changes = nparr[:-1] != nparr[1:]
        change_idxs = np.argwhere(changes==True)
        end_idxs = np.append(change_idxs + 1, len(ls))
        start_idxs = np.insert(change_idxs + 1, 0, 0)
        vals = np.take(nparr, start_idxs, axis=0)

        sparse_ls.interval_val_map = SortedDict(zip(zip(start_idxs, end_idxs), vals))

        return sparse_ls

    def _sanitized_int_idx(self, key, slice_end=False):
        if slice_end:
            if key > self.list_size:
                key = self.list_size
        else:
            if key > self.list_size - 1:
                raise IndexError
        if key < 0:
            key = self.list_size + key
        return key

    def _get_interval_from_key(self, key):
        if isinstance(key, slice):
            if key.start >= key.stop:
                raise IndexError
            return (self._sanitized_int_idx(key.start), self._sanitized_int_idx(key.stop, slice_end=True))
        elif isinstance(key, int):
            return (self._sanitized_int_idx(key), self._sanitized_int_idx(key + 1))
        else:
            raise TypeError

    def __setitem__(self, key, val):

        new_interval = self._get_interval_from_key(key)
        new_interval_start = new_interval[0]
        new_interval_end = new_interval[1]

        old_intervals = list(self.interval_val_map.keys())
        for old_interval in old_intervals:
            old_interval_start = old_interval[0]
            old_interval_end = old_interval[1]
            old_val = self.interval_val_map[old_interval]

            del self.interval_val_map[old_interval]

            if new_interval_start <= old_interval_start and new_interval_end < old_interval_end and new_interval_end > old_interval_start:
                self.interval_val_map[(new_interval_end, old_interval_end)] = old_val

            elif new_interval_start > old_interval_start and new_interval_end < old_interval_end:
                self.interval_val_map[(old_interval_start, new_interval_start)] = old_val
                self.interval_val_map[(new_interval_end, old_interval_end)] = old_val

            elif new_interval_start > old_interval_start and new_interval_start < old_interval_end and new_interval_end >= old_interval_end:
                self.interval_val_map[(old_interval_start, new_interval_start)] = old_val

            elif old_interval_end <= new_interval_start or old_interval_start >= new_interval_end:
                self.interval_val_map[old_interval] = old_val

        if val != self.fill_value:
            self.interval_val_map[new_interval] = val
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            # TODO: do this properly
            out = []
            start = self._sanitized_int_idx(key.start)
            stop = self._sanitized_int_idx(key.stop)
            step = key.step
            if not step:
                step = 1
            for i in range(start, stop, step):
                out.append(self.__getitem__(i))
            return out
        elif isinstance(key, int):
            key = self._sanitized_int_idx(key)
            i = self.interval_val_map.bisect_right((key + 1, -1))
            if not i:
                return self.fill_value
            interval = self.interval_val_map.keys()[i-1]
            if interval[0] <= key and interval[1] > key:
                return self.interval_val_map[interval]
            return self.fill_value
        else:
            print(key)
            print(type(key))
            raise TypeError

    def __iter__(self):
        self.curr_pos = 0
        return self

    def __next__(self):
        if self.curr_pos < self.list_size:
            res = self.__getitem__(self.curr_pos)
            self.curr_pos += 1
            return res
        else:
            raise StopIteration

    def _arith(self, other, op):
        if np.isscalar(other):
            self.interval_val_map = {key: op(val, other) for key, val in self.interval_val_map.items()}
            self.fill_value = op(self.fill_value, other)
        else:
            raise TypeError
            
        return self

    def __add__(self, other):
        return self._arith(other, add)

    def __sub__(self, other):
        return self._arith(other, sub)
    
    def __mul__(self, other):
        return self._arith(other, mul)

    def __len__(self):
        return self.list_size

    def __str__(self):
        return str(self.interval_val_map)