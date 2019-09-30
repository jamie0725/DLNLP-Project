# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import pickle
from torch.nn import functional as F
import torch.distributions as D

from LSTM.model import LSTMClassifier
from TextCNN.model import TextCNN
from utils.utils import print_statement, print_value
from dataset.utils import QCDataset
from torch.utils.data import DataLoader

# Some global parameters.
MODEL_LOC = 'rationale/model/best_model.pt'


class PreGenerator(nn.Module):

    def __init__(self, hidden_size, embedding_size, lstm_layer, lstm_dirc, embeddings_vector, trainable):
        super(PreGenerator, self).__init__()

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=lstm_layer,
            bidirectional=lstm_dirc,
            batch_first=True,
        )
        self.embed = nn.Embedding.from_pretrained(embeddings_vector, freeze=trainable)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = F.avg_pool1d(x, kernel_size=x.size(-1)).squeeze()
        x = self.sigmoid(x)
        return x


def dummy_task(args):
    print_statement('LOAD EMBEDDINGS')
    with open('dataset/ind2token', 'rb') as f:
        ind2token = pickle.load(f)
    with open('dataset/token2ind', 'rb') as f:
        token2ind = pickle.load(f)
    with open('dataset/embeddings_vector', 'rb') as f:
        embeddings_vector = pickle.load(f)
    print_value('Embed shape', embeddings_vector.shape)
    print_value('Vocab size', len(ind2token))

    batch_size = args.batch_size
    embedding_size = embeddings_vector.shape[1]
    qcdataset = QCDataset(token2ind, ind2token, batch_first=True)
    dataloader_train = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)
    qcdataset = QCDataset(token2ind, ind2token, split='val', batch_first=True)
    dataloader_validate = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)
    qcdataset = QCDataset(token2ind, ind2token, split='test', batch_first=True)
    dataloader_test = DataLoader(qcdataset, batch_size=args.batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)

    if args.classifier == 'LSTM':
        classifier = LSTMClassifier(
            output_size=args.num_classes,
            hidden_size=args.num_hidden,
            embedding_length=embedding_size,
            embeddings_vector=torch.from_numpy(embeddings_vector),
            lstm_layer=args.lstm_layer,
            lstm_dirc=args.lstm_bidirectional,
            trainable=args.embed_trainable,
            device=args.device
        )
        ckpt_path = 'LSTM/model/best_model.pt'

    elif args.classifier == 'TextCNN':
        classifier = TextCNN(
            batch_size=batch_size,
            c_out=args.c_out,
            output_size=args.num_classes,
            vocab_size=len(ind2token),
            embedding_size=embedding_size,
            embeddings_vector=torch.from_numpy(embeddings_vector),
            kernel_sizes=args.kernel_sizes,
            trainable=args.embed_trainable,
            p=args.p
        )
        ckpt_path = 'TextCNN/model/best_model.pt'

    ckpt = torch.load(ckpt_path, map_location=args.device)
    classifier.load_state_dict(ckpt['state_dict'])
    for parameter in classifier.parameters():
        parameter.requires_grad = False
    classifier.to(args.device)
    classifier.eval()

    pregen = PreGenerator(
        hidden_size=args.num_hidden_rationale,
        embedding_size=embedding_size,
        lstm_layer=args.lstm_layer_rationale,
        lstm_dirc=args.lstm_bidirectional_rationale,
        embeddings_vector=torch.from_numpy(embeddings_vector),
        trainable=args.embed_trainable
    )
    pregen.to(args.device)

    if args.mode == 'train':
        print_statement('MODEL TRAINING')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pregen.parameters(), lr=args.lr)
        best_eval = 0
        iteration = 0
        max_iterations = args.epochs * len(dataloader_train)
        for i in range(args.epochs):
            for batch_inputs, batch_targets in dataloader_train:
                iteration += 1
                batch_inputs = batch_inputs.to(args.device)
                batch_targets = batch_targets.to(args.device)
                pregen.train()
                optimizer.zero_grad()
                prob = pregen(batch_inputs)
                dist = D.Bernoulli(probs=prob)
                pregen_output = dist.sample()
                batch_inputs_masked = pregen_output.long() * batch_inputs
                classifier_output = classifier(batch_inputs_masked)
                selection_loss = args.lambda_1 * torch.sum(pregen_output)
                # TODO: Implement transition loss.
                classify_loss = criterion(classifier_output, batch_targets)
                generator_loss = dist.log_prob(pregen_output).sum() / len(batch_targets)
                loss = -(selection_loss.detach() - classify_loss.detach()) * generator_loss
                accuracy = float(torch.sum(classifier_output.argmax(dim=1) == batch_targets)) / len(batch_targets)
                keep = float(torch.sum(pregen_output)) / pregen_output.numel()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pregen.parameters(), max_norm=args.max_norm)
                optimizer.step()
                if iteration % 10 == 0:
                    print('Train step: {:d}/{:d}, Train loss: {:3f}, Train accuracy: {:.3f}, Keep percentage: {:.2f}'.format(iteration, max_iterations, loss, accuracy, keep))
                if iteration % 100 == 0 and iteration > 0:
                    print_statement('MODEL VALIDATING')
                    pregen.eval()
                    accs = []
                    keeps = []
                    length = []
                    elements = []
                    for batch_inputs, batch_targets in dataloader_validate:
                        batch_inputs = batch_inputs.to(args.device)
                        batch_targets = batch_targets.to(args.device)
                        with torch.no_grad():
                            prob = pregen(batch_inputs)
                            dist = D.Bernoulli(probs=prob)
                            pregen_output = dist.sample()
                            batch_inputs_masked = pregen_output.long() * batch_inputs
                            classifier_output = classifier(batch_inputs_masked)
                        acc = torch.sum(classifier_output.argmax(dim=1) == batch_targets)
                        keep = torch.sum(pregen_output)
                        length.append(len(batch_targets))
                        accs.append(acc)
                        keeps.append(keep)
                        elements.append(pregen_output.numel())
                    validate_acc = float(sum(accs)) / sum(length)
                    validate_keep = float(sum(keeps)) / sum(elements)
                    print('Testing on {} data:'.format(sum(length)))
                    print('+ Validation accuracy: {:.3f}'.format(validate_acc))
                    print('+ Keep percentage: {:.2f}'.format(validate_keep))
                    # save best model parameters
                    if validate_acc > best_eval:
                        print("New highscore! Saving model...")
                        best_eval = validate_acc
                        ckpt = {
                            "state_dict": pregen.state_dict(),
                            "optimizerizer_state_dict": optimizer.state_dict(),
                            "best_eval": best_eval
                        }
                        torch.save(ckpt, MODEL_LOC)
    else:
        print_statement('MODEL TESTING')
        raise NotImplementedError
