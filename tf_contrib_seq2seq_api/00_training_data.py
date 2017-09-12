import numpy as np
# regular expression
import re

from pprint import pprint

from collections import Counter

# maximum length of input and target sentences including paddings
enc_sentence_length = 10
dec_sentence_length = 10

# Batch_size: 2
input_batches = [
    ['Hi What is your name?', 'Nice to meet you!'],
    ['Which programming language do you use?', 'See you later.'],
    ['Where do you live?', 'What is your major?'],
    ['What do you want to drink?', 'What is your favorite beer?']]

target_batches = [
    ['Hi this is Jaemin.', 'Nice to meet you too!'],
    ['I like Python.', 'Bye Bye.'],
    ['I live in Seoul, South Korea.', 'I study industrial engineering.'],
    ['Beer please!', 'Leffe brown!']]

all_input_sentences = []
for input_batch in input_batches:
    all_input_sentences.extend(input_batch)
all_target_sentences = []
for target_batch in target_batches:
    all_target_sentences.extend(target_batch)

print (input_batches[0])
print (target_batches[0])

def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    return tokens

print (tokenizer('Hello world?? "sdfs%@#%'))

def build_vocab(sentences, is_target=False, max_vocab_size=None):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()

    for sentence in sentences:
        tokens = tokenizer(sentence)
        # build a dict of countings
        word_counter.update(tokens)
    print (word_counter)
    if max_vocab_size == None:
        max_vocab_size = len(word_counter)
    if is_target:
        vocab['_GO'] = 0
        vocab['_PAD'] = 1
        vocab_index = 2
        for key, val in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_index
            vocab_index += 1
    else:
        vocab['_PAD'] = 0
        vocab_index = 1
        for key, val in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_index
            vocab_index += 1

    for key, val in vocab.items():
        reverse_vocab[val] = key
    
    return vocab, reverse_vocab, max_vocab_size

enc_vocab, enc_reverse_vocab, enc_vocab_size = build_vocab(all_input_sentences)
dec_vocab, dec_reverse_vocab, dec_vocab_size = build_vocab(all_target_sentences, is_target=True)

def token2idx(word, vocab):
    return vocab[word]

for token in tokenizer('Nice to meet you!'):
    print(token, token2idx(token, enc_vocab))