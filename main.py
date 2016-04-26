#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import pickle

from autopoetry import AutoPoetry
from rnn import RNN

# Generate or load the training data
file = './ygz.txt'
delimiters = [':', '：', '，', ',', '。', '!', '\?', '？', '！', ' ', '　', '「', '」']
vocabulary_size = 3000
start_token = 'START'
end_token = 'END'
unknown_token = 'UNKNOWN'
data_file = 'data/training.dat'

if not os.path.exists(data_file):
    ap = AutoPoetry(file, delimiters, vocabulary_size, start_token, end_token, unknown_token)
    (X, T) = ap.get_training_data()
    with open(data_file, 'wb') as f:
        pickle.dump((X, T, ap), f)
else:
    with open(data_file, 'rb') as f:
        (X, T, ap) = pickle.load(f)

vocabulary_size = ap.vocabulary_size

# RNN training
n_features = vocabulary_size
n_hiddens = 100
epoch = 100
learning_rate = 1e-1
lr_factor = 0.9

rnn = RNN(n_features, n_hiddens, bptt_truncate=10)
rnn.train(X, T, epoch=epoch, learning_rate=learning_rate, lr_factor=lr_factor)

# Generate sentences
num_sentences = 100
senten_min_length = 3

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = ap.generate_sentence(rnn)
    print(''.join(sent))
