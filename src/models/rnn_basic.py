from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math

import numpy as np
import tensorflow as tf

# from tensorflow.python.ops import control_flow_ops
from tensorflow.models.rnn import linear
from tensorflow import control_flow_ops
# from tensorflow import linear

import pdb

batch_size = 2
num_stocks = 100
num_steps = 3

hidden_size = 50

learning_rate = 0.01
max_grad_norm = 5

def rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
  """Creates a recurrent neural network specified by RNNCell "cell".

  The simplest form of RNN network generated is:
    state = cell.zero_state(...)
    outputs = []
    states = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
      states.append(state)
    return (outputs, states)

  However, a few other options are available:

  An initial state can be provided.
  If sequence_length is provided, dynamic calculation is performed.

  Dynamic calculation returns, at time t:
    (t >= max(sequence_length)
        ? (zeros(output_shape), zeros(state_shape))
        : cell(input, state)

  Thus saving computational time when unrolling past the max sequence length.

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a vector with shape [batch_size].
    initial_state: (optional) An initial state for the RNN.  This must be
      a tensor of appropriate type and shape [batch_size x cell.state_size].
    dtype: (optional) The data type for the initial state.  Required if
      initial_state is not provided.
    sequence_length: An int64 vector (tensor) size [batch_size].
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, states) where:
      outputs is a length T list of outputs (one for each input)
      states is a length T list of states (one state following each input)

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell, RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  states = []
  with tf.variable_scope(scope or "RNN"):
    batch_size = tf.shape(inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length:  # Prepare variables
      zero_output_state = (
          tf.zeros(tf.pack([batch_size, cell.output_size]),
                   inputs[0].dtype),
          tf.zeros(tf.pack([batch_size, cell.state_size]),
                   state.dtype))
      max_sequence_length = tf.reduce_max(sequence_length)

    output_state = (None, None)
    for time, input_ in enumerate(inputs):
      if time > 0:
        tf.get_variable_scope().reuse_variables()
      output_state = cell(input_, state)
      if sequence_length:
        (output, state) = control_flow_ops.cond(
            time >= max_sequence_length,
            lambda: zero_output_state, lambda: output_state)
      else:
        (output, state) = output_state

      outputs.append(output)
      states.append(state)

    return (outputs, states)

class RNNCell(object):
  """Abstract object representing an RNN cell.

  An RNN cell, in the most abstract setting, is anything that has
  a state -- a vector of floats of size self.state_size -- and performs some
  operation that takes inputs of size self.input_size. This operation
  results in an output of size self.output_size and a new state.

  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by a super-class, MultiRNNCell,
  defined later. Every RNNCell must have the properties below and and
  implement __call__ with the following signature.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: 2D Tensor with shape [batch_size x self.input_size].
      state: 2D Tensor with shape [batch_size x self.state_size].
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:
      - Output: A 2D Tensor with shape [batch_size x self.output_size]
      - New state: A 2D Tensor with shape [batch_size x self.state_size].
    """
    raise NotImplementedError("Abstract method")

  @property
  def input_size(self):
    """Integer: size of inputs accepted by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """Integer: size of state used by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return state tensor (shape [batch_size x state_size]) filled with 0.

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      A 2D Tensor of shape [batch_size x state_size] filled with zeros.
    """
    zeros = tf.zeros(tf.pack([batch_size, self.state_size]), dtype=dtype)
    # The reshape below is a no-op, but it allows shape inference of shape[1].
    return tf.reshape(zeros, [-1, self.state_size])


class BasicRNNCell(RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = tf.tanh(linear.linear([inputs, state], self._num_units, True))
    return output, output


class BasicRNN(object):

    def __init__(self, is_training, input_data):
        self.targets = tf.placeholder(tf.float32, [batch_size, num_steps, num_stocks])
        self.input_w = tf.Variable(tf.truncated_normal([num_stocks, hidden_size],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='input_w')
        self.input_b = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
        self.output_w = tf.Variable(tf.truncated_normal([hidden_size, num_stocks],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='output_w')
        self.output_b = tf.Variable(tf.constant(0.1, shape=[num_stocks]))

        cell = BasicRNNCell(hidden_size)
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        self.input_data = tf.placeholder(tf.float32, [batch_size, num_steps, num_stocks])
        input_trans = tf.matmul(tf.reshape(self.input_data, [batch_size*num_steps, num_stocks]), self.input_w) + self.input_b
        #input_trans = tf.reshape(input_trans, [batch_size, num_steps, hidden_size])
        input_list = tf.split(0,num_steps,input_trans)

        total_loss = 0.0
        output = []
        # for b in range(batch_size):
        #     inputs = input_data[b,:,:]
        #     target = targets[b,:,:]
        outputs, self.final_states = rnn(cell, input_list, initial_state=self.initial_state)
            # output.append(outputs)


        # may need to reshape output first
        outputs = tf.concat(0, outputs)
        # pdb.set_trace()
        logits = tf.nn.xw_plus_b(outputs, self.output_w, self.output_b)
        # TODO: produce output_stocks with these variables ^
        output_stocks = tf.reshape(logits, [batch_size, num_steps, num_stocks])
        #pdb.set_trace()
        self.loss = tf.reduce_sum(tf.pow((output_stocks - self.targets),2))# / num_steps / batch_size # mean squared error

        # TODO: optimize loss
        # self._lr = tf.Variable(0.1, trainable=False)
        # tvars = tf.trainable_variables()
        # # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        # grads, _ = tf.gradients(self.loss, tvars)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.lr = tf.Variable(0.02, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                      5)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.init_op = tf.initialize_all_variables()

def get_inputs():
    # for now generate random sequences
    return np.random.rand(batch_size, num_steps+1, num_stocks)

def run():
    start = "1995-01-01"
    end = "2005-01-01"
    symbols = symbols_in_int(start, end)
    data = percent_changes(symbols)
    session = tf.InteractiveSession()
    tf.Graph().as_default()
    model = BasicRNN(True, inputs)
    session.run(model.init_op)
    total_loss = 0.0
    stock_batch = get_inputs() # TODO: replace with real data

    for i in range(1000):
        current_loss, _ = session.run([model.loss, model.train_op],
                # Initialize the LSTM state from the previous iteration.
                feed_dict={model.input_data: stock_batch[:,:-1,:], model.targets: stock_batch[:,1:,:]})
        print(current_loss)

    current_loss = session.run([model.loss], # TODO: properly define these targets
            # Initialize the LSTM state from the previous iteration.
            feed_dict={model.input_data: stock_batch[:,:-1,:], model.targets: stock_batch[:,1:,:]})
    print(current_loss)


if __name__ == "__main__":
    run()
