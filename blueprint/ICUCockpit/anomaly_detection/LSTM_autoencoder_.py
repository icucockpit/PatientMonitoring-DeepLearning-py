import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell

import numpy as np


class LSTMAutoencoder(object):

    """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)

  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

    def __init__(
        self,
        hidden_num,
        batch_size,
        window_size,
        element_num,
        reverse=True,
        decode_without_input=False,
        ):
        """
    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      batch_size : batch_size.
      window_size : the number of frames inside a datapoint
      element_num : the size of the feature vector
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
    """
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.window_size = window_size
        self.element_num = element_num

        self.input_data = tf.placeholder(tf.float32, shape=[None, self.window_size, self.element_num], name = "input")
        inputs = [tf.squeeze(t, [1]) for t in tf.split(self.input_data, self.window_size, 1)]
        cell = tf.nn.rnn_cell.LSTMCell(self.hidden_num, use_peepholes=True)
        self._enc_cell = cell
        self._dec_cell = cell

        with tf.variable_scope('encoder'):
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(
                tf.truncated_normal([self.hidden_num, self.element_num], dtype=tf.float32), name='dec_weight')
            dec_bias_ = tf.Variable(tf.constant(0.1, shape=[self.element_num], dtype=tf.float32), name='dec_bias')

            if decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32) for _ in range(len(inputs))]
                (dec_outputs, dec_state) = tf.contrib.rnn.static_rnn(
                    self._dec_cell,
                    dec_inputs,
                    initial_state=self.enc_state,
                    dtype=tf.float32
                )
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [self.batch_size, 1, 1])
                self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
            else:
                dec_state = self.enc_state
                dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
                dec_outputs = []
                for step in range(len(inputs)):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = self._dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
                    dec_outputs.append(dec_input_)
                if reverse:
                    dec_outputs = dec_outputs[::-1]
                self.output_ = tf.transpose(tf.stack(dec_outputs), [1,0, 2])

        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_), name = "loss")
