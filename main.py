# -*- coding: utf-8 -*-

import sys
import os
import argparse
import fasttext
import json
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from utilities import *
from torch.utils.data import DataLoader
from dataset.utils import *
from models.lstm import LSTMClassifier

# Some global parameters.
TRAIN_LOG_LOC = 'results/train.log'
TEST_LOG_LOC = 'results/test.log'
LABEL_JSON_LOC = 'dataset/labels.json'
TRAIN_JSON_LOC = 'dataset/train/train.json'
VAL_JSON_LOC = 'dataset/val/val.json'
TEST_JSON_LOC = 'dataset/test/test.json'
TRAIN_TXT_LOC = 'results/train.txt'
VAL_TXT_LOC = 'results/val.txt'
TEST_TXT_LOC = 'results/test.txt'
MODEL_LOC = 'results/model.bin'
EMBEDDINGS_LOC = 'GoogleNews-vectors-negative300.bin'
if __name__ == "__main__":

    # Load FastText parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='lstm',
                        help='train or eval')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_update_rate', type=float, default=100,
                        help='change the rate of updates for the learning rate')
    parser.add_argument('--dim', type=int, default=100,
                        help='size of word vectors')
    parser.add_argument('--batch_size', type=int, default=45,
                        help='batch size')
    parser.add_argument('--ws', type=int, default=1,
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
    args = parser.parse_args()

    # Create log object.
    if args.mode == 'train':
        sys.stdout = Logger(TRAIN_LOG_LOC)
    else:
        sys.stdout = Logger(TEST_LOG_LOC)

    print_statement('HYPERPARAMETER SETTING')
    print_flags(args)

    # Load data.
    print_statement('DATA PROCESSING')
    label_map = load_json(LABEL_JSON_LOC, reverse=True, name='Label Mapping')
    train_data = load_json(TRAIN_JSON_LOC, label_map, name='Training Set')
    val_data = load_json(VAL_JSON_LOC, label_map, name='Validation Set')
    test_data = load_json(TEST_JSON_LOC, label_map, name='Test Set')

    # Train model.
    if args.mode == 'train':
        # Model training.
        if args.method == 'fasttext':
            # Convert data to required file format.
            print_statement('CONVERTING DATA')
            convert_to_txt(train_data, label_map, TRAIN_TXT_LOC)
            convert_to_txt(val_data, label_map, VAL_TXT_LOC)
            convert_to_txt(test_data, label_map, TEST_TXT_LOC)
            print_statement('Done', number=0)
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
                                            )
            model.save_model(MODEL_LOC)
            print_statement('Done', number=0)
            # Testing on validation set.
            print_statement('MODEL VALIDATING')
            val_overall_result = model.test(VAL_TXT_LOC)
            val_ind_result = model.test_label(VAL_TXT_LOC)
            print_result(val_overall_result)
            print_result(val_ind_result)

        if args.method == 'lstm':
            print_statement('LOAD EMBEDDINGS')
            with open('dataset/ind2token','rb') as f:
                ind2token = pickle.load(f)
                f.close()
            with open('dataset/token2ind','rb') as f:
                token2ind = pickle.load(f)
                f.close()
            with open('dataset/embeddings_vector','rb') as f:
                embeddings_vector = pickle.load(f)
                f.close()
            print_value('embeddings_shape',embeddings_vector.shape)
            print_value('vocab_size',len(ind2token))
            input_tensor, output_tensor = convert_to_tensor(train_data,label_map,token2ind)
            print_statement('MODEL TRAINING')
            batch_size =  args.batch_size
            embedding_length = embeddings_vector.shape[1]
            vocab_size = embeddings_vector.shape[0]
            qcdataset = QCDataset(token2ind, ind2token)
            dataloader_train = DataLoader(qcdataset , batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)
            qcdataset = QCDataset(token2ind, ind2token,split='val')
            dataloader_validate = DataLoader(qcdataset , batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)
            qcdataset = QCDataset(token2ind, ind2token,split='test')
            dataloader_test = DataLoader(qcdataset , batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)
            model = LSTMClassifier(batch_size, len(label_map), 256, vocab_size, embedding_length)
            with torch.no_grad():
                model.embed.weight.data.copy_(torch.from_numpy(embeddings_vector))
                model.embed.weight.requires_grad = False
            criterion = torch.nn.CrossEntropyLoss()
            optim = torch.optim.RMSprop(model.parameters(), lr= args.lr)
            
            for step, (batch_inputs,batch_targets) in enumerate(dataloader_train):
                model.train()
                optim.zero_grad()
                output = model(batch_inputs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                loss = criterion(output, batch_targets) 
                accuracy = float(torch.sum(output.argmax(dim=1)== batch_targets)) / len(batch_targets)
                loss.backward()
                optim.step()
                if step % 10 ==0:
                    print('iter={:d},loss={:4f},acc={:.4f}'.format(step,loss,accuracy))
                # if step % 50 ==0:
                #     model.eval()
                #     accs=[]
                #     for i,(batch_inputs,batch_targets) in enumerate(dataloader_validate):
                #         output =  model(batch_inputs)
                #         acc = float(torch.sum(output.argmax(dim=1)== batch_targets)) / len(batch_targets)
                #         accs.append(acc)
                #     print(f'{step},{np.mean(accs)}')
                #     model.train()
            
    else:
        # Testing on test set.
        if args.method == 'fasttext':
            model = fasttext.load_model(MODEL_LOC)
            print_statement('MODEL TESTING')
            test_overall_result = model.test(TEST_TXT_LOC)
            test_ind_result = model.test_label(TEST_TXT_LOC)
            print_result(test_overall_result)
            print_result(test_ind_result)
