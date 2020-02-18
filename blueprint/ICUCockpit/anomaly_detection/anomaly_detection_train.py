#from anomaly_detection.LstmAutoEncoder import LstmAutoEncoder
from dataset.config_reader import ConfigReader
from dataset.DataGenerator import DataGenerator
from ICUCockpit.anomaly_detection.LSTM_autoencoder_ import LSTMAutoencoder
import argparse
import tensorflow as tf
import os
import time
import datetime
import dataset.utils
from dataset.utils import Stage
from dataset.utils import TestType
import numpy as np


class AnomalyDetection():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Lstm-based autoencoder for anomaly detection')
        parser.add_argument('-c', '--config_file', required=True, help='Configuration file with all the parameters.')
        parser.add_argument('-i', '--input_file', required=True, help='Name of the file with the features.')
        self.args = parser.parse_args()
        self.config_reader = ConfigReader()
        self.config_reader.read(self.args.config_file)
        self.input_file = self.args.input_file

    def shuffle_files(self):
        return self.args.shuffle_files

    def execute_tf(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=utils.str2bool(self.config_reader.get("allow_soft_placement")),
                log_device_placement=utils.str2bool(self.config_reader.get("log_device_placement"))
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                ae = LSTMAutoencoder(
                    self.config_reader.get("hidden_num"),
                    self.config_reader.get("batch_size"),
                    self.config_reader.get("window_size"),
                    self.config_reader.get("element_num"),
                    decode_without_input=True
                )

                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(self.config_reader.get("learning_rate"))
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(ae.loss, tvars), self.config_reader.get("max_grad_norm"))
                train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Train Summaries
                loss_summary = tf.summary.scalar("loss", ae.loss)

                #mean of loss for validation
                with tf.variable_scope("metrics"):
                    metrics = {'loss': tf.metrics.mean(ae.loss)}

                update_metrics_op = tf.group(*[op for _, op in metrics.values()])
                metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
                metrics_init_op = tf.variables_initializer(metric_variables)

                train_summary_op = tf.summary.merge([loss_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Validation summaries
                val_summary_op = tf.summary.merge([loss_summary, ])
                val_summary_dir = os.path.join(out_dir, "summaries", "dev")
                val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

                # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config_reader.get("num_checkpoints"))

                sess.run(tf.global_variables_initializer())
                sess.graph.finalize()

                # Define training and validation steps (batch)
                def train_step(inputs):
                    """
                    A single training step
                    """
                    print('global_step: %s' % tf.train.global_step(sess, global_step))
                    feed_dict = {
                        ae.input_data: inputs,
                    }
                    (loss_val, _, summaries, step) = sess.run(
                        [ae.loss, train_op, train_summary_op, global_step],
                        feed_dict
                    )
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}".format(time_str, step, loss_val))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step():
                    """
                    Evaluates model on a validation data
                    """
                    validation_generator = DataGenerator(
                        self.input_file,
                        self.config_reader.get("batch_size"),
                        Stage.validation,
                        self.config_reader.get("epochs"),
                        self.config_reader.get("train_labels"),
                        self.config_reader.get("test_labels")
                    )
                    validation_generator_data = validation_generator.generate_batches()
                    sess.run(metrics_init_op)

                    for x_val, y_val, date_time in validation_generator_data:
                        feed_dict = {
                            ae.input_data: x_val,
                        }
                        (loss_val, summaries, step, mean_val) = sess.run(
                            [
                                ae.loss,
                                val_summary_op,
                                global_step,
                                update_metrics_op
                            ],
                            feed_dict
                        )
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, {} date_time, {} label"
                            .format(time_str, step, loss_val, date_time, y_val)
                        )

                    metrics_values = {k: v[0] for k, v in metrics.items()}
                    metrics_val = sess.run(metrics_values)

                    mean_summary = tf.Summary()
                    mean_summary.value.add(tag='loss', simple_value=metrics_val["loss"])
                    val_summary_writer.add_summary(mean_summary,step)


                training_generator = DataGenerator(
                    self.input_file,
                    self.config_reader.get("batch_size"),
                    Stage.train,
                    self.config_reader.get("epochs"),
                    self.config_reader.get("train_labels"),
                    self.config_reader.get("test_labels")
                )
                training_generator_data = training_generator.generate_batches()

                for x_batch, _, _ in training_generator_data:
                    train_step(x_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.config_reader.get("checkpoint_every") == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                    if current_step % self.config_reader.get("evaluate_every") == 0:
                        print("Evaluation:\n")
                        dev_step()

                (input_, output_) = sess.run([ae.input_, ae.output_], {ae.input_data: x_batch})
                print('train result :')
                print('input :', input_[0, :, :].flatten())
                print('output :', output_[0, :, :].flatten())

if __name__ == '__main__':
    ad = AnomalyDetection()
    ad.execute_tf()
