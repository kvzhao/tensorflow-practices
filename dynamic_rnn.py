from __future__ import print_function
import tensorflow as tf

class dynamic_rnn():
    def __init__(self, cell_size, n_classes):
        self.cell_size = cell_size
        self.n_classes = n_classes

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.seqlen = tf.placeholder(dtype=tf.int32, shape=[None])

        self._build()

    def __call__(self, batch, seqlen):
        '''
            Args:
                batch: batch of input data, shape = [batch_size, num_features]
        '''
        pass
    
    def _build(self):
        print ('Building dynamic rnn...')
        with tf.variable_scope('dynamic_rnn'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, state_is_tuple=True)
            outputs, states = tf.nn.dynamic_rnn(
                inputs=self.x,
                cell=lstm_cell,
                dtype=tf.float32,
                sequence_length=self.seqlen
            )

model = dynamic_rnn(cell_size=16, n_classes=2)