from dataset.config_reader import ConfigReader
from dataset.DataGenerator import DataGenerator
import argparse
import utils
from utils import Stage
from utils import TestType

class TestLen():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Test the Lstm-based autoencoder for anomaly detection')
        parser.add_argument('-c', '--config_file', required=True, help='Configuration file with all the parameters.')
        parser.add_argument('-i', '--input_file', required=True, help='Name of the file with the features.')
        parser.add_argument(
            '-s',
            '--stage',
            required=True,
            help='Expects the stage type: 1 for train, 2 for validation, 3 for test'
        )

        self.args = parser.parse_args()
        self.config_reader = ConfigReader()
        self.config_reader.read(self.args.config_file)
        self.get_stage(self.args.stage)
        self.input_file = self.args.input_file
        self.get_test_type(self.args)

    def get_test_type(self, args):
        if utils.str2bool(args.test_type) == True:
            self.test_type = TestType.positive
        else:
            self.test_type = TestType.negative

    def get_stage(self, stage):
        if stage == 1:
            self.stage = Stage.train
        elif stage == 2:
            self.stage = Stage.validation
        elif stage == 3:
            self.stage = Stage.test
        else:
            exit()
    def execute(self):
        test_generator = DataGenerator(
            self.input_file,
            self.config_reader.get("batch_size"),
            Stage.test,
            self.config_reader.get("epochs"),
            self.config_reader.get("train_labels"),
            self.config_reader.get("test_labels"),
            self.test_type
        )

        dict = test_generator.get_class_distribution()
        for key, value in dict.items():
            print("key, value ", key, value)



if __name__ == '__main__':
    tl = TestLen()
    tl.execute()
