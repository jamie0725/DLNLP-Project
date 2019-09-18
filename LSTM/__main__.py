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
from LSTM.utils import *
from torch.utils.data import DataLoader
from dataset.utils import *
from LSTM.model import LSTMClassifier

# Some global parameters.
TRAIN_LOG_LOC = 'LSTM/results/train.log'
TEST_LOG_LOC = 'LSTM/results/test.log'
LABEL_JSON_LOC = 'dataset/labels.json'
TRAIN_JSON_LOC = 'dataset/train/train.json'
VAL_JSON_LOC = 'dataset/val/val.json'
TEST_JSON_LOC = 'dataset/test/test.json'
TRAIN_TXT_LOC = 'LSTM/results/train.txt'
VAL_TXT_LOC = 'LSTM/results/val.txt'
TEST_TXT_LOC = 'LSTM/results/test.txt'
MODEL_LOC = 'LSTM/results/model.bin'
EMBEDDINGS_LOC = 'GoogleNews-vectors-negative300.bin'
if __name__ == "__main__":

    # Load FastText parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=45,
                        help='batch size')
    
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
        