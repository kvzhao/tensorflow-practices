import numpy as np
from utils import *

class DataSampler(object):
    def __init__(self):
        self.input_batches = [
        ['Hi What is your name?', 'Nice to meet you!'],
        ['Which programming language do you use?', 'See you later.'],
        ['Where do you live?', 'What is your major?'],
        ['What do you want to drink?', 'What is your favorite beer?']]

        self.target_batches = [
            ['Hi this is Jaemin.', 'Nice to meet you too!'],
            ['I like Python.', 'Bye Bye.'],
            ['I live in Seoul, South Korea.', 'I study industrial engineering.'],
            ['Beer please!', 'Leffe brown!']]
        
        all_input_sentences = []
        all_target_sentences = []
        for input_batch in self.input_batches:
            all_input_sentences.extend(input_batch)
        for target_batch in self.target_batches:
            all_target_sentences.extend(target_batch)

        self.encoder_vocab, self.encoder_reverse_vocab, self.encoder_vocab_size = build_vocab(all_input_sentences)
        self.decoder_vocab, self.decoder_reverse_vocab, self.decoder_vocab_size = build_vocab(all_target_sentences, is_target=True)