

class BaseDatasource(object):
    def get_aggregate_timeseries(self):
        raise NotImplementedError

    def get_appliance_ids(self):
        raise NotImplementedError

    def get_appliance_id_to_label_map(self):
        raise NotImplementedError

    def get_ground_truth_disaggregated_timeseries(self):
        raise NotImplementedError

    def get_changepoints(self):
        raise NotImplementedError