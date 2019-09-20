import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from TextCNN.utils import print_statement, print_value, ClassificationTool, load_json
from dataset.utils import QCDataset


class TextCNN(nn.Module):
    def __init__(self, batch_size, c_out, output_size, vocab_size, embedding_size, embeddings_vector, kernel_sizes, trainable, p):
        super(TextCNN, self).__init__()
        self.batch_size = batch_size
        self.c_out = c_out
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # self.embed = nn.Embedding(vocab_size, embedding_size)
        self.embed = nn.Embedding.from_pretrained(embeddings_vector, freeze=trainable)
        self.kernel_sizes = kernel_sizes
        self.convolutions = nn.ModuleList([nn.Conv2d(1, self.c_out, (kernel_size, self.embedding_size)) for kernel_size in self.kernel_sizes])
        self.fc = nn.Linear(len(self.kernel_sizes) * self.c_out, self.output_size)
        self.dropout = nn.Dropout(p)
        self.Relu = nn.ReLU()

    def forward(self, x):
        # Assuming x has shape batch_size x seq_length, after embedding the shape of the output is batch_size x seq_length x embedding_size
        x = self.embed(x)
        # Expand the dimension for the conv2d layer. (batch_size x 1 x seq_length x embedding_size)
        x = x.unsqueeze(dim=1)
        # Run through the convolution layer and squeeze the height of the output feature map (dim=3) for each convolution size.
        x = [conv(x).squeeze(3) for conv in self.convolutions]
        x = [self.Relu(x_) for x_ in x]
        # Perform max pooling over time and squeeze the width (dim=2), the output shape is batch_size x c_out for each convolution size.
        x = [F.max_pool1d(x_, kernel_size=x_.size(2)).squeeze(2) for x_ in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)


def train(args, MODEL_LOC):
    print_statement('LOAD EMBEDDINGS')
    with open('dataset/ind2token', 'rb') as f:
        ind2token = pickle.load(f)
    with open('dataset/token2ind', 'rb') as f:
        token2ind = pickle.load(f)
    with open('dataset/embeddings_vector', 'rb') as f:
        embeddings_vector = pickle.load(f)
    print_value('embeddings_shape', embeddings_vector.shape)
    print_value('vocab_size', len(ind2token))
    print_statement('MODEL TRAINING')
    batch_size = args.batch_size
    embedding_size = embeddings_vector.shape[1]
    vocab_size = embeddings_vector.shape[0]
    qcdataset = QCDataset(token2ind, ind2token, batch_first=True)
    dataloader_train = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn, drop_last=True)
    qcdataset = QCDataset(token2ind, ind2token, split='val', batch_first=True)
    dataloader_validate = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn, drop_last=True)
    qcdataset = QCDataset(token2ind, ind2token, split='test', batch_first=True)
    dataloader_test = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn, drop_last=True)
    model = TextCNN(batch_size=batch_size,
                    c_out=args.c_out,
                    output_size=args.num_classes,
                    vocab_size=len(ind2token),
                    embedding_size=embedding_size,
                    embeddings_vector=torch.from_numpy(embeddings_vector),
                    kernel_sizes=args.kernel_sizes,
                    trainable=args.embed_trainable,
                    p=args.p)
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_eval = 0
    iteration = 0
    for i in np.arange(args.epochs):
        for step, (batch_inputs, batch_targets) in enumerate(dataloader_train):
            iteration += 1
            batch_inputs = batch_inputs.to(args.device)
            batch_targets = batch_targets.to(args.device)
            model.train()
            optim.zero_grad()
            output = model(batch_inputs)
            loss = criterion(output, batch_targets)
            accuracy = float(torch.sum(output.argmax(dim=1) == batch_targets)) / len(batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            optim.step()
            if iteration % 10 == 0:
                print('iter={:d}, loss={:4f}, acc={:.4f}'.format(iteration, loss, accuracy))
            
            if iteration % 100 ==0 and iteration>0:
                model.eval()
                accs=[]
                for batch_inputs,batch_targets in dataloader_validate:
                    with torch.no_grad():
                        output =  model(batch_inputs)
                    acc = float(torch.sum(output.argmax(dim=1)== batch_targets)) / len(batch_targets)
                    accs.append(acc)
                validate_acc = np.mean(accs)
                print_statement('VALIDATING')
                print('validate_accuarcy={:.4f}'.format(validate_acc))
                # save best model parameters
                if validate_acc > best_eval:
                    print("New highscore! Saving model...")
                    best_eval = validate_acc
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizerizer_state_dict": optim.state_dict(),
                        "best_eval": best_eval,
                    }
                    torch.save(ckpt, MODEL_LOC)
                model.train()


def test(args, MODEL_LOC, LABEL_JSON_LOC):
    print_statement('LOAD EMBEDDINGS')
    label_map = load_json(LABEL_JSON_LOC, reverse=True, name='Label Mapping')

    with open('dataset/ind2token', 'rb') as f:
        ind2token = pickle.load(f)
    with open('dataset/token2ind', 'rb') as f:
        token2ind = pickle.load(f)
    with open('dataset/embeddings_vector', 'rb') as f:
        embeddings_vector = pickle.load(f)
    print_value('embeddings_shape', embeddings_vector.shape)
    print_value('vocab_size', len(ind2token))
    batch_size = args.batch_size
    embedding_size = embeddings_vector.shape[1]
    vocab_size = embeddings_vector.shape[0]
    qcdataset = QCDataset(token2ind, ind2token, batch_first=True)
    dataloader_train = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn, drop_last=True)
    qcdataset = QCDataset(token2ind, ind2token, split='val', batch_first=True)
    dataloader_validate = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn, drop_last=True)
    qcdataset = QCDataset(token2ind, ind2token, split='test', batch_first=True)
    dataloader_test = DataLoader(qcdataset, batch_size=batch_size, shuffle=True, collate_fn=qcdataset.collate_fn, drop_last=True)

    model = TextCNN(batch_size=batch_size,
                c_out=args.c_out,
                output_size=args.num_classes,
                vocab_size=len(ind2token),
                embedding_size=embedding_size,
                embeddings_vector=torch.from_numpy(embeddings_vector),
                kernel_sizes=args.kernel_sizes,
                trainable=args.embed_trainable,
                p=args.p)
    model.to(args.device)
    ckpt = torch.load(MODEL_LOC)
    model.load_state_dict(ckpt["state_dict"])
    print_statement('MODEL TESTING')
    qcdataset = QCDataset(token2ind, ind2token, split='test')
    dataloader_test = DataLoader(qcdataset, batch_size=args.batch_size, shuffle=True, collate_fn=qcdataset.collate_fn)
    ct = ClassificationTool(len(label_map))
    accs=[]
    for batch_inputs,batch_targets in dataloader_validate:
        with torch.no_grad():
            output =  model(batch_inputs)
        acc = float(torch.sum(output.argmax(dim=1)== batch_targets)) / len(batch_targets)
        accs.append(acc)
        ct.update(output,batch_targets)
    test_acc = np.mean(accs)
    print('Overall ACC {:.4f}'.format(test_acc))
    PREC,REC,F1 = ct.get_result()
    for i,classname in enumerate(label_map.keys()): 
        print('* {} PREC: {:.2f}, REC: {:.2f}, F1: {:.2f}'.format(classname, PREC[i], REC[i],F1[i]))
