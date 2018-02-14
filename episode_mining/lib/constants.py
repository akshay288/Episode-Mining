import os


REPOSITORY_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..{}'.format(os.sep) * 2)

DATA_DIR = os.path.join(REPOSITORY_ROOT_DIR, 'DATA/')

BLUED_DATA_DIR = os.path.join(DATA_DIR, 'BLUED/')
BLUED_GROUND_TRUTH_DATA_DIR = os.path.join(DATA_DIR, 'BLUED_GroundTruth/')

REDD_DATA_DIR = os.path.join(DATA_DIR, 'REDD/')
ECOLABS_DATA_DIR = os.path.join(DATA_DIR, 'ecolabs/')

K_MEANS_DIR = os.path.join(REPOSITORY_ROOT_DIR, 'genetic_k_means/')

ZERO_THRESH = 30
SAME_CP_THRESH = 120

class Colls(object):
    POWER = 'power'
    REACT = 'react'

    POWER_DIFFS = 'power_diff'
    REACT_DIFFS = 'react_diff'

    APP_ID = 'appid'
    APP_LABEL = 'applabel'
    PHASE = 'phase'
    LOCATION = 'location'

    CLUSTER_NUM = 'cluster_num'