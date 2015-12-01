from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn.ptb import reader
from tensorflow.models.rnn import rnn

batch_size = 1
num_stocks = 100
num_steps = 3

hidden_size = 50

class BasicRNN(object):

    def __init__(self, is_training, input_data, input_targets):
        self._targets = input_targets
        self._input_data = input_data

        cell = rnn_cell.BasicRNNCell(hidden_size)
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        logits = tf.nn.xw_plus_b(output, tf.get_variable("output_w", [hidden_size, num_stocks]),
                                 tf.get_variable("output_b", [num_stocks]))

        total_loss = 0.0
        output = []
        for b in range(batch_size):
            inputs = self._input_data[b]
            targets = self._targets[b]
            outputs, states = rnn.rnn(cell, inputs, initial_state=self.intial_state)
            output.append(outputs)

        logits = tf.nn.xw_plus_b(output, tf.get_variable("output_w", [hidden_size, num_stocks]),
                                 tf.get_variable("output_b", [num_stocks]))
        # TODO: produce output_stocks with these variables ^

        loss = tf.reduce_sum((output_stocks - self._targets)**2) / num_steps / batch_size # mean squared error

        # TODO: optimize loss 



def get_inputs():
    # for now generate random sequences
    return np.random.rand(batch_size, num_steps+1, num_stocks)

def run():

    inputs = get_inputs()
    session = tf.InteractiveSession()
    tf.Graph().as_default()

    # A numpy array holding the state of LSTM after each batch of words.
    numpy_state = initial_state.eval()
    total_loss = 0.0
    for current_batch_of_words in words_in_dataset:
        numpy_state, current_loss = session.run([final_state, loss],
            # Initialize the LSTM state from the previous iteration.
            feed_dict={initial_state: numpy_state, stocks: current_batch_of_stocks})
        total_loss += current_loss

if __name__ == "__main__":
    run()
