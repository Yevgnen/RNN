#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
import itertools
import re

import jieba
import nltk
import scipy as sp


class AutoPoetry(object):
    def __init__(self, file, delimiters, vocabulary_size, start_token, end_token, unknown_token):
        self.file = file
        self.delimiters = delimiters
        self.vocabulary_size = vocabulary_size
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token

    def read_file(self):
        data_file = codecs.open(self.file, 'r', 'utf-8')
        sentences = []

        while 1:
            line = data_file.readline()

            if not line:
                break

            if line.startswith('title') or line == '\r\n':
                continue

            line = line.strip()
            # There may be multiple sentences in a line
            sents = re.split('|'.join(self.delimiters), line)
            # Add start and end tokens
            sents = ['{0} {1} {2}'.format(self.start_token, sent, self.end_token) for sent in sents if len(sent) > 0]
            sentences.append(sents)

        sentences = list(itertools.chain(*sentences))

        print('Read {0} sentences.'.format(len(sentences)))

        return sentences

    def get_training_data(self):
        sentences = self.read_file()
        tokenized_sentences = [list(jieba.cut(sent, cut_all=False)) for sent in sentences]
        tokenized_sentences = [[word for word in sent if word not in set(self.delimiters)]
                               for sent in tokenized_sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        if len(word_freq.items()) < self.vocabulary_size:
            self.vocabulary_size = len(word_freq.items())
        print("Found {0} unique words tokens.".format(len(word_freq.items())))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(self.vocabulary_size - 1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(self.unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        print("Using vocabulary size {0}.".format(self.vocabulary_size))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else self.unknown_token
                                      for w in sent]

        # print("Example sentence after Pre-processing: {0}".format(tokenized_sentences[0]))

        # Create the training data
        X_train = sp.asarray([[word_to_index[w] for w in sent[:-1]]
                              for sent in tokenized_sentences])
        y_train = sp.asarray([[word_to_index[w] for w in sent[1:]]
                              for sent in tokenized_sentences])

        self.index_to_word = index_to_word
        self.word_to_index = word_to_index

        return (X_train, y_train)

    def generate_sentence(self, model):
        # We start the sentence with the start token
        new_sentence = [self.word_to_index[self.start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == self.word_to_index[self.end_token]:
            next_word_probs = model.forward_probability(new_sentence)
            sampled_word = self.word_to_index[self.unknown_token]
            # We don't want to sample unknown words
            while sampled_word == self.word_to_index[self.unknown_token]:
                samples = sp.random.multinomial(1, next_word_probs)
                sampled_word = sp.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [self.index_to_word[x] for x in new_sentence[1:-1]]

        return sentence_str
