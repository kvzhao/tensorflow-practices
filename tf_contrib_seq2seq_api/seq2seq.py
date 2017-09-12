import math

from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils import *

from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops.helper import TrainingHelper
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder, BasicDecoderOutput

class Seq2SeqModel(object):
    def __init__ (self, config, data, mode='train'):
        assert mode in ['train', 'test', 'inference']
        self.mode = mode
        self.vocab_size = config.vocab_size
        # Data preprocess
        self.data = data

        # Network
        self.hidden_size = config.hidden_size    
        self.encoder_embed_size = config.encoder_embed_size
        self.decoder_embed_size = config.decoder_embed_size
        self.cell = config.cell

        # Training
        self.optimizer = config.optimizer
        self.n_epoch = config.n_epoch
        self.learning_rate = config.learning_rate
        
        self.ckpt_dir = config.ckpt_dir

        self._build_graph()
    
    def _build_graph(self):
        print ('Start building graph...')
        self._init_placeholder()
        self._init_encoder()
        self._init_decoder()
        print ('... Graph is built.')
    
    def _init_placeholder(self):
        print ('Initialize placeholders')
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32, 
            shape=[None, None],
            name='input_sequence'
        )
        self.encoder_sequence_length = tf.placeholder(
            dtype=tf.int32,
            shape=[None,],
            name='input_sequence_length'
        )
        if self.mode == 'train':
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='target_sequences'
            )

            self.decoder_sequence_length = tf.placeholder(
                dtype=tf.int32,
                shape=[None,],
                name='target_sequence_length'
            )

    def _init_encoder(self):
        print ('Initialize Encoder')
        with tf.variable_scope('Encoder') as scope:
            # Embedding
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            self.encoder_embed = tf.get_variable(
                name='embedding',
                initializer=initializer,
                shape=[self.vocab_size+1, self.encoder_embed_size],
                dtype=tf.float32
            )

            # [Batch_size x encoder_sentence_length x embedding_size]
            encoder_embed_inputs = tf.nn.embedding_lookup(
                self.encoder_embed, self.encoder_inputs, name='embed_inputs'
            )
            encoder_cell = self.cell(self.hidden_size)

            # encoder_outputs: [batch_size x encoder_sentence_length x embedding_size]
            # encoder_last_state: [batch_size x embedding_size]
            encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=encoder_embed_inputs,
                sequence_length=self.encoder_sequence_length,
                time_major=False,
                dtype=tf.float32)
    
    def _init_decoder(self):
        print ('Initialize Decoder')
        with tf.variable_scope('Decoder') as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            self.decoder_embed = tf.get_variable(
                name='embedding',
                initializer=initializer,
                shape=[self.vocab_size+2, self.decoder_embed_size],
                dtype=tf.float32
            )
            decoder_cell = self.cell(self.hidden_size)

            output_layer = Dense(self.vocab_size+2, name='output_projection')

            if self.mode == 'train':

                max_decode_length = tf.reduce_max(self.decoder_sequence_length+1, name='max_decoder_length')

                decoder_embed_inputs = tf.nn.embedding_lookup(
                    self.decoder_embed, self.decoder_inputs, name='embed_inputs')

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=decoder_embed_inputs,
                    sequence_length=self.decoder_sequence_length+1,
                    time_major=False,
                    name='train_helper')

                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=training_helper,
                    initial_state=self.encoder_last_state,
                    output_layer=output_layer
                )

                #API:https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode
                training_decoder_outputs, training_decoder_last_state, last_length = tf.contrib.seq2seq.dynamic_decode(
                    training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decode_length
                )

                # logits: [batch_size x max_dec_len x dec_vocab_size+2]
                logits = tf.identity(training_decoder_outputs.rnn_output, name='logits')
                targets = tf.slice(self.decoder_inputs, [0, 0], [-1, max_decode_length], name='targets')
                # masks: [batch_size x max_dec_len]
                # => ignore outputs after `decoder_senquence_length+1` when calculating loss
                masks = tf.sequence_mask(self.decoder_sequence_length+1, max_decode_length, 
                                        dtype=tf.float32, name='masks'
                )

                # API:https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
                # internal: `tf.nn.sparse_softmax_cross_entropy_with_logits`
                self.batch_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=logits,
                    targets=targets,
                    weights=masks,
                    name='batch_loss'
                )

                # prediction sample for validation
                self.valid_predictions = tf.identity(training_decoder_outputs.sample_id, name='valid_preds')

                # list of trainable weights
                self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            elif self.mode == 'inference':
                pass

    def _init_training_operator(self):
        self.train_op = self.optimizer(self.learning_rate, name='train_op').minimize(self.batch_loss)
    
    def train(self, sess, from_scratch=True, load_ckpt=None, save_path=None):
        if self.mode == 'train':
            # Restore checkpoints
            # TODO
            # Add Optimizer 
            self._init_training_operator()
            # Initail Weights
            sess.run(tf.global_variables_initializer())

            # DATA
            input_batches, target_batches = self.data.input_batches, self.data.target_batches
            loss_history = []
            
            for epoch in tqdm(range(self.n_epoch)):
                all_preds=[]
                epoch_loss=0.0

                for input_batch, target_batch in zip(input_batches, target_batches):
                    input_batch_tokens=[]
                    target_batch_tokens=[]
                    encoder_sentence_length=[]
                    decoder_sentence_length=[]

                    for input_sentence in input_batch:
                        tokens, sentence_length = sentence2idx(input_sentence, 
                                                            self.data.encoder_vocab,
                                                            max_sentence_length=10)
                        input_batch_tokens.append(tokens)
                        encoder_sentence_length.append(sentence_length)
                    for target_sentence in target_batch:
                        tokens, sentence_length = sentence2idx(target_sentence, 
                                                            self.data.decoder_vocab,
                                                            max_sentence_length=10,
                                                            is_target=True)
                        target_batch_tokens.append(tokens)
                        decoder_sentence_length.append(sentence_length)

                    # Evaluation operations
                    batch_preds, batch_loss, _ = sess.run(
                        [self.valid_predictions, self.batch_loss, self.train_op],
                        feed_dict={
                            self.encoder_inputs: input_batch_tokens,
                            self.encoder_sequence_length: encoder_sentence_length,
                            self.decoder_inputs: target_batch_tokens,
                            self.decoder_sequence_length: decoder_sentence_length
                        })

                    epoch_loss += batch_loss
                    all_preds.append(batch_preds)

                loss_history.append(epoch_loss)

                # Logging every 400 epochs
                if epoch % 100 == 0:
                    print('Epoch', epoch)
                    for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                        for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                            print('\tInput: {}'.format(input_sent))
                            print('\tPrediction: {}'.format(idx2sentence(pred, reverse_vocab=self.data.decoder_reverse_vocab)))
                            print('\tTarget:, {}'.format(target_sent))
                    print('\tepoch loss: {}\n'.format(epoch_loss))
