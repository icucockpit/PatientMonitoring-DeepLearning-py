import argparse
from enum import Enum

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Stage(Enum):
    train = 1
    validation = 2
    test = 3

class TestType(Enum):
    positive = 1
    negative = 2