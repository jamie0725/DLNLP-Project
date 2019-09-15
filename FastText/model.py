# -*- coding: utf-8 -*-

import sys
import os
import argparse
import fasttext
import json

# Some global parameters.
TRAIN_LOG_LOC = 'train.log'
TEST_LOG_LOC = 'test.log'
LABEL_JSON_LOC = '../dataset/labels.json'
TRAIN_JSON_LOC = '../dataset/train/train.json'
VAL_JSON_LOC = '../dataset/val/val.json'
TEST_JSON_LOC = '../dataset/test/test.json'
TRAIN_TXT_LOC = 'train.txt'
VAL_TXT_LOC = 'val.txt'
TEST_TXT_LOC = 'test.txt'
MODEL_LOC = 'model.bin'


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


def load_json(file_loc, mapping=None, verbose=2, reverse=False, name=None):
    '''
    Load json file at a given location.
    '''

    with open(str(file_loc), 'r') as file:
        data = json.load(file)
        file.close()
    if reverse:
        data = dict(map(reversed, data.items()))
    if verbose > 0:
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

    if args.verbose > 0:
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
    else:
        pass


def print_flags():
    """
    Print all entries in args variable.
    """

    if args.verbose > 0:
        for key, value in vars(args).items():
            print(key + ' : ' + str(value))


def print_result(result, keep=3):
    """
    Print result matrix.
    """

    if type(result) == dict:
        # Individual result.
        for key in result:
            prec = result[key]['precision']
            rec = result[key]['recall']
            f1 = result[key]['f1score']
            key = key.replace('__label__', '')[:keep]
            print('* {} PREC: {:.2f}, {} REC: {:.2f}, {} F1: {:.2f}'.format(key, prec, key, rec, key, f1))
    elif type(result) == tuple:
        # Overall result.
        print('Testing on {} data:'.format(result[0]))
        print('+ Overall ACC: {:.3f}'.format(result[1]))
        assert result[1] == result[2]
    else:
        raise TypeError


def convert_to_txt(data, label, file_loc):
    '''
    Convert data to fasttext training format.
    '''

    with open(file_loc, mode='w', encoding='utf-8') as file:
        for key in data:
            name = '__label__' + label[int(key)]
            for query in data[key]:
                file.write('{}\t{}\n'.format(name, ' '.join(query)))
        file.close()


if __name__ == "__main__":

    # Load FastText parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_update_rate', type=float, default=100,
                        help='change the rate of updates for the learning rate')
    parser.add_argument('--dim', type=int, default=100,
                        help='size of word vectors')
    parser.add_argument('--ws', type=int, default=5,
                        help='size of the context window')
    parser.add_argument('--epoch', type=int, default=5,
                        help='train or eval')
    parser.add_argument('--min_count', type=int, default=1,
                        help='minimal number of word occurences')
    parser.add_argument('--neg', type=int, default=5,
                        help='number of negatives sampled')
    parser.add_argument('--word_ngrams', type=int, default=1,
                        help='max length of word ngram')
    parser.add_argument('--char_ngrams', type=int, default=0,
                        help='max length of character ngram')
    parser.add_argument('--loss', type=str, default='softmax',
                        help='loss function (softmax, ns, hs)')
    parser.add_argument('--verbose', type=int, default=2,
                        help='silent: 0, progress bar: 1, detailed: 2')
    args = parser.parse_args()

    # Create log object.
    if args.mode == 'train':
        sys.stdout = Logger(TRAIN_LOG_LOC)
    else:
        sys.stdout = Logger(TEST_LOG_LOC)

    print_statement('HYPERPARAMETER SETTING')
    print_flags()

    # Load data.
    print_statement('DATA PROCESSING')
    label_map = load_json(LABEL_JSON_LOC, reverse=True, name='Label Mapping', verbose=args.verbose)
    train_data = load_json(TRAIN_JSON_LOC, label_map, name='Training Set', verbose=args.verbose)
    val_data = load_json(VAL_JSON_LOC, label_map, name='Validation Set', verbose=args.verbose)
    test_data = load_json(TEST_JSON_LOC, label_map, name='Test Set', verbose=args.verbose)

    # Train model.
    if args.mode == 'train':
        # Convert data to required file format.
        print_statement('CONVERTING DATA')
        convert_to_txt(train_data, label_map, TRAIN_TXT_LOC)
        convert_to_txt(val_data, label_map, VAL_TXT_LOC)
        convert_to_txt(test_data, label_map, TEST_TXT_LOC)
        print_statement('Done', number=0)

        # Model training.
        print_statement('MODEL TRAINING')
        model = fasttext.train_supervised(input=TRAIN_TXT_LOC,
                                          lr=args.lr,
                                          dim=args.dim,
                                          ws=args.ws,
                                          epoch=args.epoch,
                                          minCount=args.min_count,
                                          maxn=args.char_ngrams,
                                          neg=args.neg,
                                          wordNgrams=args.word_ngrams,
                                          loss=args.loss,
                                          bucket=100000,
                                          lrUpdateRate=args.lr_update_rate,
                                          verbose=args.verbose)
        model.save_model(MODEL_LOC)
        print_statement('Done', number=0)

        # Testing on validation set.
        print_statement('MODEL VALIDATING')
        val_overall_result = model.test(VAL_TXT_LOC)
        val_ind_result = model.test_label(VAL_TXT_LOC)
        print_result(val_overall_result)
        print_result(val_ind_result)
    else:
        # Testing on test set.
        model = fasttext.load_model(MODEL_LOC)
        print_statement('MODEL TESTING')
        test_overall_result = model.test(TEST_TXT_LOC)
        test_ind_result = model.test_label(TEST_TXT_LOC)
        print_result(test_overall_result)
        print_result(test_ind_result)
