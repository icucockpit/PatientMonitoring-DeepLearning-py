from dataset.config_reader import ConfigReader
from dataset.DataGenerator import DataGenerator
from ICUCockpit.anomaly_detection.LSTM_autoencoder_ import LSTMAutoencoder
import argparse
import tensorflow as tf
import utils
from dataset.utils import Stage
from dataset.utils import TestType
import os
import time

class AnomalyDetectionTest():

    def __init__(self):
        parser = argparse.ArgumentParser(description='Test the Lstm-based autoencoder for anomaly detection')
        parser.add_argument('-c', '--config_file', required=True, help='Configuration file with all the parameters.')
        parser.add_argument('-i', '--input_file', required=True, help='Name of the file with the features.')
        parser.add_argument('-m', '--model', required=True, help='Checkpoint of the model')
        parser.add_argument('-t', '--test_type', required=True, help='Test type: boolean, on positive or negative data')
        self.args = parser.parse_args()
        self.config_reader = ConfigReader()
        self.config_reader.read(self.args.config_file)
        self.input_file = self.args.input_file
        self.checkpoint_file_path = self.args.model
        self.get_test_type(self.args)

    def get_test_type(self, args):
        if utils.str2bool(args.test_type) == True:
            self.test_type = TestType.positive
        else:
            self.test_type = TestType.negative

    def execute_tf(self):
        test_generator = DataGenerator(
            self.input_file,
            self.config_reader.get("batch_size"),
            Stage.test,
            self.config_reader.get("epochs"),
            self.config_reader.get("train_labels"),
            self.config_reader.get("test_labels"),
            self.test_type
        )
        test_generator_data = test_generator.generate_batches()
        checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_file_path)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=utils.str2bool(self.config_reader.get("allow_soft_placement")),
                log_device_placement=utils.str2bool(self.config_reader.get("log_device_placement"))
             )
            sess = tf.Session(config=session_conf)

            with sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                inputs = graph.get_operation_by_name("input").outputs[0]
                loss = graph.get_operation_by_name("loss").outputs[0]

                loss_summary = tf.summary.scalar("loss", loss)

                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                test_summary_op = tf.summary.merge([loss_summary])
                test_summary_dir = os.path.join(out_dir, "summaries", "test")
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

                i = 0
                for x_data, _, date_time in test_generator_data:
                    feed_dict = {
                         inputs: x_data,
                    }
                    (test_loss, summary) = sess.run([loss, test_summary_op], feed_dict)
                    test_summary_writer.add_summary(summary,i)
                    i = i+1
                    print("l", test_loss, date_time)

                test_summary_writer.close()

if __name__ == '__main__':
    adt = AnomalyDetectionTest()
    adt.execute_tf()
