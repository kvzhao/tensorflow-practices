'''
    This work is from https://github.com/ematvey/ai-copywriter/blob/master/model.py
    here is informative introduction to seq2seq http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
'''

import tensorflow as tf
import math
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse, embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, GRUCell, LSTMStateTuple

class Seq2SeqModel():
    # special tokens
    PAD = 0
    EOS = 1

    def __init__ (self, encoder_cell, decoder_cell, 
                        vocab_size, embedding_size,
                        bidirectional=False,
                        attention=False):
        self.bidirectional = bidirectional
        self.attention = attention

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self._build_graph()
    
    def _build_graph(self):
        # allocate placeholders
        self._init_placeholder()
        # init embeddings, connectors
        self._init_decoder_train_connectors()
        self._init_embeddings()
        # init encoders
        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()
        # init decoders
        self._init_decoder()
        # declare optimizer
        self._init_optimizer()
    
    def _init_placeholder(self):
        # time major
        self.encoder_inputs = tf.placeholder(
            shape = (None, None),
            dtype=tf.int32,
            name ='encoder_inputs'
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )
        # only used in training
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length'
        )

    def _init_decoder_train_connectors(self):
        '''
            decoder 
        '''
        with tf.name_scope('DecoderTrainFeed'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            # Put EOS in inputs
            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            # different with decoder_targets, complicated preprocessing
            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            # WAHT this mask do? ==> Answered below.
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_targets_length-1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            # transpose to be time-major
            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [0, 1])

            # hacky way using one_hot to put EOS symbol at the end of target sequence
            decoder_train_targets = tf.add(decoder_train_targets, 
                                            decoder_train_targets_eos_mask)
            self.decoder_train_targets = decoder_train_targets

            self.loss_weight = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name='loss_weights')

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:

            # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)

            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)
    
    def _init_simple_encoder(self):
            with tf.variable_scope("Encoder") as scope:
                (self.encoder_outputs, self.encoder_state) = (
                    tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                        inputs=self.encoder_inputs_embedded,
                                        sequence_length=self.encoder_inputs_length,
                                        time_major=True,
                                        dtype=tf.float32)
                    )


    def _init_bidirectional_encoder(self):
        with tf.variable_scope("BidirectionalEncoder") as scope:
            ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_state, encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32))

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, LSTMStateTuple):
                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')
    
    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
                )
            else:
                pass
                # Attention

                # attention states: shape = [batch_size, max_time, num_units]
                '''
                    ci = \sum^n_{ij}\alpha_{ij}h_j 
                    i: generated word
                    j: time step
                    n: seq length
                '''
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

                (attention_keys,
                attention_values,
                attention_score_fn,
                attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
                )
                # end of if attention 

            (self.decoder_outputs_train, self.decoder_state_train, self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=True,
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')
            scope.reuse_variables()

            (self.decoder_logits_inference, self.decoder_state_inference, self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=True,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')
            

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets, weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)