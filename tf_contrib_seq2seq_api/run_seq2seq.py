import tensorflow as tf
import numpy as np

from data_sampler import DataSampler
from democonfig import DemoConfig
from seq2seq import Seq2SeqModel


with tf.Session() as sess:
    config = DemoConfig()
    data = DataSampler()
    model = Seq2SeqModel(config, data)

    model.train(sess)