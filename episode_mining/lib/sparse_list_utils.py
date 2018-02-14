from interval import interval


def find_sparse_list(sparse_ls, pred, default=0):
    """
    keeps the parts in a sparse list that match a predicate, everything else
    is set to default
    find_sparse_list(sparse_ls, lambda x: x > 5) -> sparse_ls but only with values
    greater than 5
    """
    set_sparse_list(sparse_ls, lambda e: not pred(e), default)


def set_sparse_list(sparse_ls, pred, new_val):
    """
    sets all elements in a sparse list that match a predicate to new_val
    """
    if pred(sparse_ls.fill_value):
        sparse_ls.fill_value = new_val
    for key, val in sparse_ls.interval_val_map.items():
        if pred(val):
            sparse_ls.interval_val_map[key] = new_val


def get_interval_intersect(intervals_a, intervals_b):
    # convert to inclusive ends
    intervals_a = [(start, end - 1) for start, end in intervals_a]
    intervals_b = [(start, end - 1) for start, end in intervals_b]

    inclusive_intersect = list(interval(*intervals_a) & interval(*intervals_b))

    # convert back to exclusive ends
    return [(int(start), int(end) + 1) for start, end in inclusive_intersect]


def get_interval_not(intervals, interval_range):
    # sort intervals by start
    intervals = sorted(intervals, key=lambda i: i[0])

    assert intervals[0][0] >= interval_range[0]
    assert intervals[-1][1] <= interval_range[1]

    not_intervals = []

    if intervals[0][0] != interval_range[0]:
        not_intervals.append((interval_range[0], intervals[0][0]))

    for i in range(len(intervals[:-1])):
        not_intervals.append((intervals[i][1], intervals[i+1][0]))

    if intervals[-1][1] != interval_range[1]:
        not_intervals.append((intervals[-1][1], interval_range[1]))

    return not_intervals