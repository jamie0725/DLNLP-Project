# -*- coding: utf-8 -*-

import sys
import argparse
import fasttext
from utils.utils import print_statement, print_flags, load_json, convert_to_txt, print_result, Logger

# Some global parameters.
TRAIN_LOG_LOC = 'FastText/results/train.log'
TEST_LOG_LOC = 'FastText/results/test.log'
LABEL_JSON_LOC = 'dataset/labels.json'
TRAIN_JSON_LOC = 'dataset/train/train.json'
VAL_JSON_LOC = 'dataset/val/val.json'
TEST_JSON_LOC = 'dataset/test/test.json'
TRAIN_TXT_LOC = 'FastText/train.txt'
VAL_TXT_LOC = 'FastText/val.txt'
TEST_TXT_LOC = 'FastText/test.txt'
MODEL_LOC = 'FastText/model/model.bin'


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
    parser.add_argument('--epoch', type=int, default=70,
                        help='number of epochs')
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

    print_statement('HYPERPARAMETER SETTING', verbose=args.verbose)
    print_flags(args, verbose=args.verbose)

    # Load data.
    print_statement('DATA PROCESSING', verbose=args.verbose)
    label_map = load_json(LABEL_JSON_LOC, reverse=True, name='Label Mapping', verbose=args.verbose)
    train_data = load_json(TRAIN_JSON_LOC, label_map, name='Training Set', verbose=args.verbose)
    val_data = load_json(VAL_JSON_LOC, label_map, name='Validation Set', verbose=args.verbose)
    test_data = load_json(TEST_JSON_LOC, label_map, name='Test Set', verbose=args.verbose)

    # Train model.
    if args.mode == 'train':
        # Convert data to required file format.
        print_statement('CONVERTING DATA', verbose=args.verbose)
        convert_to_txt(train_data, label_map, TRAIN_TXT_LOC)
        convert_to_txt(val_data, label_map, VAL_TXT_LOC)
        convert_to_txt(test_data, label_map, TEST_TXT_LOC)
        print_statement('Done', number=0, verbose=args.verbose)

        # Model training.
        print_statement('MODEL TRAINING', verbose=args.verbose)
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
        print_statement('Done', number=0, verbose=args.verbose)

        # Testing on validation set.
        print_statement('MODEL VALIDATING', verbose=args.verbose)
        val_overall_result = model.test(VAL_TXT_LOC)
        val_ind_result = model.test_label(VAL_TXT_LOC)
        print_result(val_overall_result, label_map)
        print_result(val_ind_result, label_map)
    else:
        # Testing on test set.
        model = fasttext.load_model(MODEL_LOC)
        print_statement('MODEL TESTING', verbose=args.verbose)
        test_overall_result = model.test(TEST_TXT_LOC)
        test_ind_result = model.test_label(TEST_TXT_LOC)
        print_result(test_overall_result, label_map)
        print_result(test_ind_result, label_map)
