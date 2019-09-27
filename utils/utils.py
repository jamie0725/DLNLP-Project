import sys
import os
import json
import re
import numpy as np
import torch
from nltk.stem.porter import PorterStemmer


class Logger(object):
    '''
    Export print to logs.
    '''

    def __init__(self, file_loc):
        self.terminal = sys.stdout
        self.log = open(file_loc, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class InputParser(object):

    def __init__(self, token2ind):
        self.token2ind = token2ind
        self.stemmer = PorterStemmer()

    def word2id(self, word):
        if word in self.token2ind.keys():
            return self.token2ind[word]
        else:
            return 0

    def sentence2id(self, word_list):
        sentence = self.stemmer.stem(' '.join(word_list))
        sentence_id = [self.word2id(word) for word in self.clean_text(sentence).split()]
        return sentence_id

    def clean_text(self, text):

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text


class ClassificationTool(object):
    """Computes PRE, REC and F1"""

    def __init__(self, output_length):
        self.output_length = output_length
        self.reset()

    def reset(self):
        n = self.output_length
        self.tp = np.zeros(n)
        self.tn = np.zeros(n)
        self.fp = np.zeros(n)
        self.fn = np.zeros(n)
        self.acc = np.zeros(n)
        self.pre = np.zeros(n)
        self.rec = np.zeros(n)
        self.f1 = np.zeros(n)

    def update(self, output, target):
        for cls in range(self.output_length):
            pred = output.argmax(dim=1) == cls
            truth = target == cls
            n_pred = ~pred
            n_truth = ~truth
            self.tp[cls] += pred.mul(truth).sum()
            self.tn[cls] += n_pred.mul(n_truth).sum()
            self.fp[cls] += pred.mul(n_truth).sum()
            self.fn[cls] += n_pred.mul(truth).sum()

    def get_result(self):
        for cls in range(self.output_length):
            self.acc[cls] = (self.tp[cls] + self.tn[cls]).sum() / (self.tp[cls] + self.tn[cls] + self.fp[cls] + self.fn[cls]).sum()
            self.pre[cls] = self.tp[cls] / (self.tp[cls] + self.fp[cls])
            self.rec[cls] = self.tp[cls] / (self.tp[cls] + self.fn[cls])
            self.f1[cls] = (2.0 * self.tp[cls]) / (2.0 * self.tp[cls] + self.fp[cls] + self.fn[cls])
        return self.pre, self.rec, self.f1


def load_json(file_loc, mapping=None, reverse=False, name=None):
    '''
    Load json file at a given location.
    '''

    with open(str(file_loc), 'r') as file:
        data = json.load(file)
        file.close()
    if reverse:
        data = dict(map(reversed, data.items()))
    print_logs(data, mapping, name)
    return data


def print_logs(data, mapping, name, keep=3):
    '''
    Print detailed information of given data iterator.
    '''

    print('Statistics of {}:'.format(name))
    try:
        count = 0
        for key in data:
            print('* Number of data in class {}: {}'.format(mapping[int(key)][:keep], len(data[key])))
            count += len(data[key])
        print('+ Total: {}'.format(count))
    except:
        print('* Class: {}'.format(data))
        print('+ Total: {}'.format(len(data)))


def print_statement(statement, isCenter=False, symbol='=', number=15, newline=False):
    '''
    Print required statement in a given format.
    '''

    if newline:
        print()
    if number > 0:
        prefix = symbol * number + ' '
        suffix = ' ' + symbol * number
        statement = prefix + statement + suffix
    if isCenter:
        print(statement.center(os.get_terminal_size().columns))
    else:
        print(statement)


def print_flags(args):
    """
    Print all entries in args variable.
    """

    for key, value in vars(args).items():
        print(key + ' : ' + str(value))


def print_value(name, value):
    print(name + f': {value}')


def convert_to_tensor(data, label_map, token2ind):
    parser = InputParser(token2ind)
    inputs = []
    outputs = []
    for i in range(len(label_map)):
        for line in data[str(i)]:
            inputs.append(parser.sentence2id(line))
            outputs.append(str(i))
    max_length = np.max([len(line) for line in inputs])
    input_tensor = torch.ones(len(inputs), max_length)
    output_tensor = torch.zeros(len(inputs))
    for i, sentence in enumerate(inputs):
        output_tensor[i] = int(outputs[i])
        for j, value in enumerate(sentence):
            input_tensor[i][j] = value
    return input_tensor, output_tensor
