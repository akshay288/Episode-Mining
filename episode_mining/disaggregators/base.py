

class BaseDisaggregator(object):
    def __init__(self, datasource, config):
        self.config = config
        self.datasource = datasource

    def get_disaggregated_timeseries(self):
        raise NotImplementedError