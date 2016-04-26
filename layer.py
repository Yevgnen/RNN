#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp

from activation import Sigmoid, Softmax, Tanh


class RNNLayer(object):
    def __init__(self, activation='tanh'):
        self.activation = activation
        self.activation_map = {'tanh': Tanh(), 'sigmoid': Sigmoid(), 'softmax': Softmax()}
        self.a = None
        self.h = None
        self.y = None

    def activate(self, x):
        return self.activation_map[self.activation].eval(x)

    def backward(self):
        return self.activation_map[self.activation].gradient(self.a)

    def loss(self, t):
        return self.activation_map[self.activation].loss(t, self.y)


class HiddenLayer(RNNLayer):
    def __init__(self, activation='tanh'):
        super(HiddenLayer, self).__init__(activation)

    def forward(self, U, x, W, prev_h, b):
        self.a = U[:, x] + sp.dot(W, prev_h) + b
        self.h = self.activate(self.a)


class OutputLayer(RNNLayer):
    def __init__(self, activation='softmax'):
        super(OutputLayer, self).__init__(activation)

    def forward(self, V, h, c):
        self.a = sp.dot(V, h) + c
        self.y = self.activate(self.a)
