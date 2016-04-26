#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rnn import RNN

x = [0, 4, 2, 5, 7]
t = [4, 2, 5, 7, 1]
n_hiddens = 4
n_features = 10
rnn = RNN(n_features, n_hiddens)
rnn.forward_propagation(x)
rnn.check_gradient(x, t)
rnn.train([x], [t], 100)
