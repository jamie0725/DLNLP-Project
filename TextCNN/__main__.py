import argparse
import torch
from TextCNN.model import train

# Some global parameters.
TRAIN_LOG_LOC = 'TextCNN/results/train.log'
TEST_LOG_LOC = 'TextCNN/results/test.log'
TRAIN_TXT_LOC = 'TextCNN/results/train.txt'
VAL_TXT_LOC = 'TextCNN/results/val.txt'
TEST_TXT_LOC = 'TextCNN/results/test.txt'
MODEL_LOC = 'TextCNN/results/model.bin'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of examples to process in a batch')
    parser.add_argument('--num_classes', type=int, default=6, help='number of target classes')
    parser.add_argument('--max_norm', type=float, default=5.0, help='max norm of gradient')
    parser.add_argument('--embed_trainable', type=bool, default=False, help='finetune pre-trained embeddings')
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[2, 3, 4], help='kernel sizes for the convolution layer')
    parser.add_argument('--device', type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--p', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--c_out', type=int, default=2, help='output channel size of the convolution layer')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
