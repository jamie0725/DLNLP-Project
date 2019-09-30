# _*_ coding: utf-8 _*_

import argparse
import torch
import sys

from rationale.model import dummy_task
from utils.utils import print_statement, print_flags, Logger

# Some global parameters.
TRAIN_LOG_LOC = 'rationale/results/train.log'
TEST_LOG_LOC = 'rationale/results/test.log'


def main():
    # Load parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, default='TextCNN', help='classifier to use "LSTM/TextCNN"')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of examples to process in a batch')
    parser.add_argument('--max_norm', type=float, default=5.0, help='max norm of gradient')
    parser.add_argument('--embed_trainable', type=bool, default=False, help='finetune pre-trained embeddings')
    parser.add_argument('--device', type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    # rationale specific parameters.
    parser.add_argument('--num_hidden_rationale', type=int, default=64, help='number of hidden units for the PreGenerator LSTM for rationale')
    parser.add_argument('--lstm_layer_rationale', type=int, default=1, help='number of layers for the PreGenerator LSTM for rationale')
    parser.add_argument('--lstm_bidirectional_rationale', type=bool, default=False, help='bi-direction for the PreGenerator LSTM for rationale')
    parser.add_argument('--lambda_1', type=float, default=2e-4, help='regularizer of the length of selected words')
    parser.add_argument('--lambda_2', type=float, default=4e-4, help='regularizer of the local coherency of words')
    # LSTM specific parameters.
    parser.add_argument('--num_hidden', type=int, default=256, help='number of hidden units in the LSTM classifier')
    parser.add_argument('--lstm_layer', type=int, default=2, help='number of layers of lstm')
    parser.add_argument('--lstm_bidirectional', type=bool, default=True, help='bi-direction of lstm')
    # TextCNN specific parameters.
    parser.add_argument('--num_classes', type=int, default=6, help='number of target classes')
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[2, 3, 4], help='kernel sizes for the convolution layer')
    parser.add_argument('--p', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--c_out', type=int, default=32, help='output channel size of the convolution layer')

    args = parser.parse_args()

    # Create log object.
    if args.mode == 'train':
        sys.stdout = Logger(TRAIN_LOG_LOC)
    else:
        sys.stdout = Logger(TEST_LOG_LOC)

    print_statement('HYPERPARAMETER SETTING')
    print_flags(args)

    dummy_task(args)


if __name__ == '__main__':
    main()
