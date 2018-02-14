

class BaseMetric(object):
    def __init__(self, disaggregated_ts, ground_truth_ts, config):
        self.disaggregated_ts = disaggregated_ts
        self.ground_truth_ts = ground_truth_ts
        self.config = config

    def get_true_positives(self):
        raise NotImplementedError
    
    def get_false_positives(self):
        raise NotImplementedError
    
    def get_false_negatives(self):
        raise NotImplementedError