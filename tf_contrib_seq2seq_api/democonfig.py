import tensorflow as tf

class DemoConfig:
    # Dataset
    vocab_size = 28
        
    # Model
    hidden_size = 30
    encoder_embed_size = 30
    decoder_embed_size = 30
    cell = tf.contrib.rnn.BasicLSTMCell
    
    # Training
    optimizer = tf.train.RMSPropOptimizer
    n_epoch = 801
    learning_rate = 0.001

    # Checkpoint Path
    ckpt_dir = './ckpt_dir/'