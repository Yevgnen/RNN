#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from datetime import datetime

import scipy as sp
from scipy import random as sprd

from layer import HiddenLayer, OutputLayer


class RNN(object):
    def __init__(self, n_features, n_hiddens, bptt_truncate=4):
        self.n_features = n_features    # The size of the dictionary.
        self.n_hiddens = n_hiddens

        # Initialize the hidden layer
        self.U = sprd.uniform(-sp.sqrt(1. / n_features), sp.sqrt(1. / n_features), (n_hiddens, n_features))
        self.W = sprd.uniform(-sp.sqrt(1. / n_hiddens), sp.sqrt(1. / n_hiddens), (n_hiddens, n_hiddens))
        self.b = sp.zeros(self.n_hiddens)

        # Initialize the output layer
        self.V = sprd.uniform(-sp.sqrt(1. / n_hiddens), sp.sqrt(1. / n_hiddens), (n_features, n_hiddens))
        self.c = sp.zeros(self.n_features)

        self.bptt_truncate = bptt_truncate

    def save(self, file='pickle.dat'):
        data = (self.n_features, self.n_hiddens, self.U, self.W, self.b, self.V, self.c, self.bptt_truncate)
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def load(self, file='pickle.dat'):
        with open(file, 'rb') as f:
            (self.n_features, self.n_hiddens,
             self.U, self.W, self.b, self.V, self.c,
             self.bptt_truncate) = pickle.load(f)

    def compute_loss(self, x, t):
        """Compute loss of a sample."""
        loss = 0.0
        cells = self.forward_propagation(x)
        for i, cell in enumerate(cells):
            one_hot_t = sp.zeros(self.n_features)
            one_hot_t[t[i]] = 1
            loss += cell[-1].loss(one_hot_t)
        return loss

    def compute_total_loss(self, X, T):
        """Compute the total loss of all samples."""
        loss = 0.0
        n_samples = len(X)
        for i in range(n_samples):
            loss += self.compute_loss(X[i], T[i])

        return loss / n_samples

    def forward_probability(self, x):
        return self.forward_propagation(x)[-1][-1].y

    def forward_propagation(self, x):
        """Forward Progation of a single sample."""
        tau = len(x)
        prev_h = sp.zeros(self.n_hiddens)

        cells = [None for i in range(tau)]
        for i in range(tau):
            # Compute the hidden state
            time_input = x[i]
            hidden = HiddenLayer()
            hidden.forward(self.U, time_input, self.W, prev_h, self.b)

            # Compute the output
            prev_h = hidden.h
            output = OutputLayer()
            output.forward(self.V, hidden.h, self.c)

            cells[i] = (hidden, output)
        return cells

    def bptt(self, x, t):
        """Back propagation throuth time of a sample.

        Reference: [1] Deep Learning, Ian Goodfellow, Yoshua Bengio and Aaron Courville, P385.
        """
        dU = sp.zeros_like(self.U)
        dW = sp.zeros_like(self.W)
        db = sp.zeros_like(self.b)
        dV = sp.zeros_like(self.V)
        dc = sp.zeros_like(self.c)

        tau = len(x)
        cells = self.forward_propagation(x)

        dh = sp.zeros(self.n_hiddens)
        for i in range(tau - 1, -1, -1):
            # FIXME:
            # 1. Should not use cell[i] since there maybe multiple hidden layers.
            # 2. Using exponential family as output should not be specified.
            time_input = x[i]
            one_hot_t = sp.zeros(self.n_features)
            one_hot_t[t[i]] = 1

            # Cell of time i
            cell = cells[i]
            # Hidden layer of current cell
            hidden = cell[0]
            # Output layer of current cell
            output = cell[1]
            # Hidden layer of time i + 1
            prev_hidden = cells[i - 1][0] if i - 1 >= 0 else None
            # Hidden layer of time i - 1
            next_hidden = cells[i + 1][0] if i + 1 < tau else None

            # Error of current time i
            da = hidden.backward()
            next_da = next_hidden.backward() if next_hidden is not None else sp.zeros(self.n_hiddens)
            prev_h = prev_hidden.h if prev_hidden is not None else sp.zeros(self.n_hiddens)

            # FIXME: The error function should not be specified here
            # do = sp.dot(output.backward().T, -one_hot_t / output.y)
            do = output.y - one_hot_t
            dh = sp.dot(sp.dot(self.W.T, sp.diag(next_da)), dh) + sp.dot(self.V.T, do)

            # Gradient back propagation through time
            dc += do
            db += da * dh
            dV += sp.outer(do, hidden.h)
            dW += sp.outer(da * dh, prev_h)
            dU[:, time_input] += da * dh

        return (dU, dW, db, dV, dc)

    def sgd_step(self, x, t, learning_rate):
        """Process SGD using one single sample."""
        (dU, dW, db, dV, dc) = self.bptt(x, t)
        self.U -= learning_rate * dU
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        self.V -= learning_rate * dV
        self.c -= learning_rate * dc

    def train(self, X, T, epoch=100, learning_rate=1e-2, lr_factor=0.9):
        """Train the network by SGD."""
        losses = sp.zeros(epoch)
        for j in range(epoch):
            # Scan the full training set
            for i, (x, t) in enumerate(zip(X, T)):
                self.sgd_step(x, t, learning_rate)

            timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            losses[j] = self.compute_total_loss(X, T)
            print('{0}: After epoch={1}, loss={2}, lr={3}.'.format(timestr, j + 1, losses[j], learning_rate))

            # Adjust the learning rate if the loss increased
            if j > 0 and losses[j] > losses[j - 1]:
                learning_rate *= lr_factor

            # Save params of each epoch
            self.save('data/epoch{0}.dat'.format(j + 1))

    def numerical_gradient(self, x, t, eps=1e-10):
        (U, W, b, V, c) = (self.U, self.W, self.b, self.V, self.c)

        dU = sp.zeros_like(U)
        dW = sp.zeros_like(W)
        db = sp.zeros_like(b)
        dV = sp.zeros_like(V)
        dc = sp.zeros_like(c)

        length = c.shape[0]
        for i in range(length):
            ci = self.c[i]
            self.c[i] = ci + eps
            lh = self.compute_loss(x, t)
            self.c[i] = ci - eps
            lo = self.compute_loss(x, t)
            dc[i] = (lh - lo) / (2.0 * eps)
            self.c[i] = ci

        (row, col) = U.shape
        for i in range(row):
            for j in range(col):
                uij = self.U[i, j]
                self.U[i, j] = uij + eps
                lh = self.compute_loss(x, t)
                self.U[i, j] = uij - eps
                lo = self.compute_loss(x, t)
                dU[i, j] = (lh - lo) / (2.0 * eps)
                self.U[i, j] = uij

        (row, col) = W.shape
        for i in range(row):
            for j in range(col):
                wij = self.W[i, j]
                self.W[i, j] = wij + eps
                lh = self.compute_loss(x, t)
                self.W[i, j] = wij - eps
                lo = self.compute_loss(x, t)
                dW[i, j] = (lh - lo) / (2.0 * eps)
                self.W[i, j] = wij

        length = b.shape[0]
        for i in range(length):
            bi = self.b[i]
            self.b[i] = bi + eps
            lh = self.compute_loss(x, t)
            self.b[i] = bi - eps
            lo = self.compute_loss(x, t)
            db[i] = (lh - lo) / (2.0 * eps)
            self.b[i] = bi

        (row, col) = V.shape
        for i in range(row):
            for j in range(col):
                vij = self.V[i, j]
                self.V[i, j] = vij + eps
                lh = self.compute_loss(x, t)
                self.V[i, j] = vij - eps
                lo = self.compute_loss(x, t)
                dV[i, j] = (lh - lo) / (2.0 * eps)
                self.V[i, j] = vij

        return (dU, dW, db, dV, dc)

    def check_gradient(self, x, t):
        (dU, dW, db, dV, dc) = self.bptt(x, t)
        (ndU, ndW, ndb, ndV, ndc) = self.numerical_gradient(x, t, 1e-5)
        print('Check gradient of bptt: max|dU-dU|={0}'.format((sp.absolute(dU - ndU)).max()))
        print('Check gradient of bptt: max|dW-dW|={0}'.format((sp.absolute(dW - ndW)).max()))
        print('Check gradient of bptt: max|db-db|={0}'.format((sp.absolute(db - ndb)).max()))
        print('Check gradient of bptt: max|dV-dV|={0}'.format((sp.absolute(dV - ndV)).max()))
        print('Check gradient of bptt: max|dc-dc|={0}'.format((sp.absolute(dc - ndc)).max()))
