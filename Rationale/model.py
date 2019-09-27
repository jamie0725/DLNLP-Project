import torch
import torch.nn as nn
import pickle
from torch.nn import functional as F

from LSTM.model import LSTMClassifier
from TextCNN.model import TextCNN
from utils.utils import print_statement, print_value
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

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = F.avg_pool1d(x, kernel_size=x.size(-1)).squeeze()
        return torch.sigmoid(x)


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
    # TODO: Maybe set the LSTM module to have batch_first=True to be consistent with TextCNN and the Rationale module.
    qcdataset = QCDataset(token2ind, ind2token, batch_first=True)
    dataloader_train = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)
    qcdataset = QCDataset(token2ind, ind2token, split='val', batch_first=True)
    dataloader_validate = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)

    # TODO: When saving checkpoints, save the classifer as well to avoid reinstantiation of the classifier classes.
    # For example:
    # checkpoint = {'model': LSTMClassifier(...), ...}
    if args.classifier == 'LSTM':
        classifier = LSTMClassifier(
            output_size=args.num_classes,
            hidden_size=args.num_hidden,
            embedding_length=embedding_size,
            embeddings_vector=torch.from_numpy(embeddings_vector),
            lstm_layer=args.lstm_layer,
            lstm_dirc=args.lstm_bidirectional,
            trainable=args.embed_trainable,
            # TODO: Maybe remove the model.to(device) inside.
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

    ckpt = torch.load(ckpt_path)
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

    for batch_inputs, batch_targets in dataloader_train:
        batch_inputs = batch_inputs.to(args.device)
        batch_targets = batch_targets.to(args.device)
        print(batch_inputs.size())
        with torch.no_grad():
            pregen_output = pregen(batch_inputs)
        print(pregen_output)
        print(pregen_output.size())
        break
