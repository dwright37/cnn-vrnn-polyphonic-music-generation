import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
import collections

ConditionalGRUState = collections.namedtuple('ConditionalGRUCellState', ['h', 'c'])

class ConditionalGRU(rnn_cell_impl.RNNCell):
    def __init__(self, 
                 num_units, 
                 reuse=False,
                 activation=None,
                 kernel_initializer=None, 
                 bias_initializer=None):

        super(ConditionalGRU, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None
    
    @property
    def state_size(self):
        #State is h and c (conditioning vec)
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype=tf.float32):
        h = tf.zeros([batch_size, self._num_units], dtype=dtype)
        c = tf.zeros([batch_size, self._num_units], dtype=dtype)

        return ConditionalGRUState(h=h, c=c)

    def call(self, inputs, state):
        """
            Conditionl GRU operations

            inputs: [batch_size, num_units]
            state: (h=[batch_size, num_units], c=[batch_size, num_units])

            output: [batch_size, num_units]
            new_state: (h=[batch_size, num_units], c=[batch_size, num_units])
        """

        h = state.h
        c = state.c

        bias_ones = self._bias_initializer
        if self._bias_initializer is None:
            bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
        with vs.variable_scope('gates'):
            val_concat = rnn_cell_impl._linear(
                                [inputs, h, c], 
                                2*self._num_units,
                                bias=False,
                                bias_initializer=self._bias_initializer,
                                kernel_initializer=self._kernel_initializer)

        val = math_ops.sigmoid(val_concat)
        r, z = array_ops.split(value=val, num_or_size_splits=2, axis=1)

        r_state = r * h

        with vs.variable_scope('candidate'):
            hbar_out = rnn_cell_impl._linear(
                                    [inputs, r_state, c],
                                    self._num_units,
                                    bias=False,
                                    bias_initializer=self._bias_initializer,
                                    kernel_initializer=self._kernel_initializer)

        hbar = self._activation(hbar_out)
        output = (1 - z) * h + z * hbar

        new_state = ConditionalGRUState(h=output, c=c)

        return output, new_state
