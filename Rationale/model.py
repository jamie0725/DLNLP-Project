# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.distributions as D
from torch.nn import functional as F

from LSTM.model import LSTMClassifier
from TextCNN.model import TextCNN
from utils.utils import print_statement, print_value, ClassificationTool, load_json
from dataset.utils import QCDataset
from torch.utils.data import DataLoader


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


def train(args, GEN_MODEL_LOC, LSTM_MODEL_LOC, TCN_MODEL_LOC):
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
        ENC_MODEL_LOC = LSTM_MODEL_LOC

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
        ENC_MODEL_LOC = TCN_MODEL_LOC
    if args.pretrained:
        ckpt = torch.load(ckpt_path, map_location=args.device)
        classifier.load_state_dict(ckpt['state_dict'])
    # for parameter in classifier.parameters():
    # parameter.requires_grad = False
    classifier.to(args.device)
    # classifier.eval()

    pregen = PreGenerator(
        hidden_size=args.num_hidden_rationale,
        embedding_size=embedding_size,
        lstm_layer=args.lstm_layer_rationale,
        lstm_dirc=args.lstm_bidirectional_rationale,
        embeddings_vector=torch.from_numpy(embeddings_vector),
        trainable=args.embed_trainable
    )
    pregen.to(args.device)

    print_statement('MODEL TRAINING')
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    gen_optimizer = torch.optim.Adam(pregen.parameters(), lr=args.lr_gen)
    enc_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr_enc)
    best_eval = 0
    iteration = 0
    max_iterations = args.epochs * len(dataloader_train)
    for i in range(args.epochs):
        for batch_inputs, batch_targets in dataloader_train:
            iteration += 1
            batch_inputs = batch_inputs.to(args.device)
            batch_targets = batch_targets.to(args.device)
            pregen.train()
            classifier.train()
            gen_optimizer.zero_grad()
            enc_optimizer.zero_grad()
            p_z_x = pregen(batch_inputs)
            dist = D.Bernoulli(probs=p_z_x)
            pregen_output = dist.sample()
            batch_inputs_masked = batch_inputs.clone()
            batch_inputs_masked[torch.eq(pregen_output, 0.)] = 1
            classifier_output = classifier(batch_inputs_masked)
            selection_loss = args.lambda_1 * pregen_output.sum(dim=-1)
            transition_loss = args.lambda_2 * (pregen_output[:, 1:] - pregen_output[:, :-1]).abs().sum(dim=-1)
            classify_loss = criterion(classifier_output, batch_targets)
            cost = selection_loss + transition_loss + classify_loss
            enc_loss = (selection_loss + transition_loss + classify_loss).mean()
            enc_loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=args.max_norm)
            enc_optimizer.step()
            gen_loss = (cost.detach() * -dist.log_prob(p_z_x).sum(dim=-1)).mean()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(pregen.parameters(), max_norm=args.max_norm)
            gen_optimizer.step()
            accuracy = float(torch.sum(classifier_output.argmax(dim=1) == batch_targets)) / len(batch_targets)
            keep = compute_keep_rate(batch_inputs, pregen_output)
            if iteration % 10 == 0:
                print('Train step: {:d}/{:d}, GEN Train loss: {:.3f}, ENC Train loss: {:.3f}, '
                      'Train accuracy: {:.3f}, Keep percentage: {:.2f}'.format(iteration, max_iterations, gen_loss, enc_loss, accuracy, keep))
            if iteration % 100 == 0 and iteration > 0:
                print_statement('MODEL VALIDATING')
                pregen.eval()
                accs = []
                keeps = []
                length = []
                elements = []
                org_pads = []
                pads_kept = []
                for batch_inputs, batch_targets in dataloader_validate:
                    batch_inputs = batch_inputs.to(args.device)
                    batch_targets = batch_targets.to(args.device)
                    with torch.no_grad():
                        p_z_x = pregen(batch_inputs)
                        dist = D.Bernoulli(probs=p_z_x)
                        pregen_output = dist.sample()
                        batch_inputs_masked = batch_inputs.clone()
                        batch_inputs_masked[torch.eq(pregen_output, 0.)] = 1
                        classifier_output = classifier(batch_inputs_masked)
                    acc = torch.sum(classifier_output.argmax(dim=1) == batch_targets)
                    keep = torch.sum(pregen_output)
                    org_pad = torch.eq(batch_inputs, 1).sum()
                    num_pads_kept = (torch.eq(batch_inputs, 1) == torch.eq(pregen_output, 1.)).sum()
                    length.append(len(batch_targets))
                    accs.append(acc)
                    keeps.append(keep)
                    org_pads.append(org_pad)
                    pads_kept.append(num_pads_kept)
                    elements.append(pregen_output.numel())
                validate_acc = float(sum(accs)) / sum(length)
                validate_keep = float(sum(keeps) - sum(pads_kept)) / float(sum(elements) - sum(org_pads))
                extract_rationale(batch_inputs, batch_inputs_masked, ind2token, validate_acc, validate_keep, args.classifier)
                print('Testing on {} data:'.format(sum(length)))
                print('+ Validation accuracy: {:.3f}'.format(validate_acc))
                print('+ Keep percentage: {:.2f}'.format(validate_keep))
                # save best model parameters
                if validate_acc > best_eval:
                    print("New highscore! Saving model...")
                    best_eval = validate_acc
                    gen_ckpt = {
                        "state_dict": pregen.state_dict(),
                        "optimizer_state_dict": gen_optimizer.state_dict(),
                        "best_eval": best_eval
                    }
                    torch.save(gen_ckpt, GEN_MODEL_LOC)
                    enc_ckpt = {
                        "state_dict": classifier.state_dict(),
                        "optimizer_state_dict": enc_optimizer.state_dict(),
                        "best_eval": validate_keep
                    }
                    torch.save(enc_ckpt, ENC_MODEL_LOC)


def test(args, GEN_MODEL_LOC, LSTM_MODEL_LOC, TCN_MODEL_LOC, LABEL_JSON_LOC):
    print_statement('LOAD EMBEDDINGS')
    label_map = load_json(LABEL_JSON_LOC, reverse=True, name='Label Mapping')
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
        ckpt_path = LSTM_MODEL_LOC

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
        ckpt_path = TCN_MODEL_LOC

    ckpt = torch.load(ckpt_path, map_location=args.device)
    classifier.load_state_dict(ckpt['state_dict'])
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
    ckpt = torch.load(GEN_MODEL_LOC, map_location=args.device)
    pregen.load_state_dict(ckpt['state_dict'])
    pregen.to(args.device)
    pregen.eval()

    print_statement('MODEL TESTING')

    qcdataset = QCDataset(token2ind, ind2token, split='test', batch_first=True)
    dataloader_test = DataLoader(qcdataset, batch_size=args.batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)

    ct = ClassificationTool(len(label_map))
    accs = []
    keeps = []
    length = []
    elements = []
    org_pads = []
    pads_kept = []

    for batch_inputs, batch_targets in dataloader_test:
        batch_inputs = batch_inputs.to(args.device)
        batch_targets = batch_targets.to(args.device)
        with torch.no_grad():
            p_z_x = pregen(batch_inputs)
            dist = D.Bernoulli(probs=p_z_x)
            pregen_output = dist.sample()
            batch_inputs_masked = batch_inputs.clone()
            batch_inputs_masked[torch.eq(pregen_output, 0.)] = 1
            classifier_output = classifier(batch_inputs_masked)
        acc = torch.sum(classifier_output.argmax(dim=1) == batch_targets)
        keep = torch.sum(pregen_output)
        org_pad = torch.eq(batch_inputs, 1).sum()
        num_pads_kept = (torch.eq(batch_inputs, 1) == torch.eq(pregen_output, 1.)).sum()
        accs.append(acc)
        keeps.append(keep)
        org_pads.append(org_pad)
        pads_kept.append(num_pads_kept)
        elements.append(pregen_output.numel())
        length.append(len(batch_targets))
        ct.update(classifier_output, batch_targets)
    test_acc = float(np.sum(accs)) / sum(length)
    test_keep = float(np.sum(keeps) - np.sum(pads_kept)) / float(sum(elements) - np.sum(org_pads))
    extract_rationale(batch_inputs, batch_inputs_masked, ind2token, test_acc, test_keep, args.classifier)
    print('Testing on {} data:'.format(sum(length)))
    print('+ Overall ACC: {:.3f}'.format(test_acc))
    print('+ Overall KEEP: {:.3f}'.format(test_keep))
    PREC, REC, F1 = ct.get_result()
    for i, classname in enumerate(label_map.values()):
        print('* {} PREC: {:.3f}, {} REC: {:.3f}, {} F1: {:.3f}'.format(classname[:3], PREC[i], classname[:3], REC[i], classname[:3], F1[i]))


def extract_rationale(batch_inputs, batch_rationale, ind2token, acc, keep, classifier):
    batch_size = batch_inputs.shape[0]
    picked = np.random.choice(batch_size, size=min(5, batch_size), replace=False)
    # inputs = batch_inputs[picked, np.arange(batch_inputs.shape[1])].tolist()
    inputs = batch_inputs[picked, :].tolist()
    # rationale = batch_rationale[picked, np.arange(batch_inputs.shape[1])].tolist()
    rationale = batch_rationale[picked, :].tolist()
    with open('Rationale/results/samples.txt', 'w') as f:
        f.write('* Classifier: {}\n'.format(classifier))
        f.write('-------------------------------------\n')
    for (input_sentence, input_rationale) in zip(inputs, rationale):
        with open('Rationale/results/samples.txt', 'a') as f:
            f.write('+ Original input:\n')
            f.write(' '.join(list(filter(lambda x: x != '<pad>', map(lambda x: ind2token[x], input_sentence)))) + '\n')
            f.write('- Extracted rationale:\n')
            f.write(' '.join(list(filter(lambda x: x != '<pad>', map(lambda x: ind2token[x], input_rationale)))) + '\n')

    with open('Rationale/results/samples.txt', 'a') as f:
        f.write('-------------------------------------\n')
        f.write('* Accuracy: {:.3f}\n'.format(acc))
        f.write('* Keep percentage: {:.3f}'.format(keep))


def compute_keep_rate(batch_inputs, pregen_output):
    num_pads_kept = (torch.eq(batch_inputs, 1) == torch.eq(pregen_output, 1.)).sum()
    num_org_pads = torch.eq(batch_inputs, 1).sum()
    return float(torch.sum(pregen_output) - num_pads_kept) / float(pregen_output.numel() - num_org_pads)
