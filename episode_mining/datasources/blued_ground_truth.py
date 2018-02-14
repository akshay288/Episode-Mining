from episode_mining.datasources.blued_base import BLUEDBaseDatasource
from episode_mining.lib.constants import Colls


class BLUEDGroundTruthDatasource(BLUEDBaseDatasource):
    def __init__(self, phase):
        super().__init__(phase)
    
    def get_changepoints(self):
        return self.blued_ground_truth[[Colls.APP_ID, Colls.LOCATION, Colls.POWER_DIFFS, Colls.REACT_DIFFS]]