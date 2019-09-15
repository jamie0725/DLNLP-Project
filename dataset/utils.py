import os
import glob
import json
import random
import numpy as np
import torch.utils.data as data
import torch
from collections import OrderedDict
from nltk.stem.porter import PorterStemmer
from preprocessing.utils import Embeddings

NUM_SAMPLE = 2000

cat2id = {
    'ABBREVIATION': 0,
    'ENTITY': 1,
    'DESCRIPTION': 2,
    'HUMAN': 3,
    'LOCATION': 4,
    'NUMERIC': 5
}

id2cat = {v: k for k, v in cat2id.items()}

all_classes = ['ABBREVIATION', 'ENTITY', 'DESCRIPTION', 'HUMAN', 'LOCATION', 'NUMERIC']

data_path = os.path.dirname(os.path.realpath(__file__))

train_file = '/train/train.json'
val_file = '/val/val.json'
test_file = '/test/test.json'


def parse_dataset():
    with open(data_path + '/labels.json', 'w') as json_file:
        json.dump(cat2id, json_file, indent=2)

    abbreviation = []
    entity = []
    description = []
    human = []
    location = []
    numeric = []

    for lbl_file in glob.iglob(data_path + '/train/*.label'):
        with open(lbl_file) as fw:
            for line in fw:
                words = line.split()
                category = words.pop(0)
                if 'ABBR' in category:
                    abbreviation.append(words)
                elif 'ENTY' in category:
                    entity.append(words)
                elif 'DESC' in category:
                    description.append(words)
                elif 'HUM' in category:
                    human.append(words)
                elif 'LOC' in category:
                    location.append(words)
                elif 'NUM' in category:
                    numeric.append(words)

    # print(len(abbreviation))
    # print(len(entity))
    # print(len(description))
    # print(len(human))
    # print(len(location))
    # print(len(numeric))

    train = {}
    val = {}
    train_portion = 0.9

    # Shuffle the data.
    random.seed(42)
    random.shuffle(abbreviation)
    random.shuffle(entity)
    random.shuffle(description)
    random.shuffle(human)
    random.shuffle(location)
    random.shuffle(numeric)

    # Truncate the data.
    abbreviation = abbreviation[:200]
    entity = entity[:NUM_SAMPLE]
    description = description[:NUM_SAMPLE]
    human = human[:NUM_SAMPLE]
    location = location[:NUM_SAMPLE]
    numeric = numeric[:NUM_SAMPLE]

    # Split into training set and validation set.
    train[cat2id['ABBREVIATION']] = abbreviation[:int(200 * train_portion)]
    train[cat2id['ENTITY']] = entity[:int(NUM_SAMPLE * train_portion)]
    train[cat2id['DESCRIPTION']] = description[:int(NUM_SAMPLE * train_portion)]
    train[cat2id['HUMAN']] = human[:int(NUM_SAMPLE * train_portion)]
    train[cat2id['LOCATION']] = location[:int(NUM_SAMPLE * train_portion)]
    train[cat2id['NUMERIC']] = numeric[:int(NUM_SAMPLE * train_portion)]

    val[cat2id['ABBREVIATION']] = abbreviation[int(200 * train_portion):]
    val[cat2id['ENTITY']] = entity[int(NUM_SAMPLE * train_portion):]
    val[cat2id['DESCRIPTION']] = description[int(NUM_SAMPLE * train_portion):]
    val[cat2id['HUMAN']] = human[int(NUM_SAMPLE * train_portion):]
    val[cat2id['LOCATION']] = location[int(NUM_SAMPLE * train_portion):]
    val[cat2id['NUMERIC']] = numeric[int(NUM_SAMPLE * train_portion):]

    with open(data_path + train_file, 'w') as json_file:
        json.dump(train, json_file, indent=2)

    with open(data_path + val_file, 'w') as json_file:
        json.dump(val, json_file, indent=2)

    test = {}
    test[cat2id['ABBREVIATION']] = []
    test[cat2id['ENTITY']] = []
    test[cat2id['DESCRIPTION']] = []
    test[cat2id['HUMAN']] = []
    test[cat2id['LOCATION']] = []
    test[cat2id['NUMERIC']] = []
    with open(data_path + '/test/TREC_10.label') as lbl_file:
        for line in lbl_file:
            words = line.split()
            category = words.pop(0)
            if 'ABBR' in category:
                test[cat2id['ABBREVIATION']].append(words)
            elif 'ENTY' in category:
                test[cat2id['ENTITY']].append(words)
            elif 'DESC' in category:
                test[cat2id['DESCRIPTION']].append(words)
            elif 'HUM' in category:
                test[cat2id['HUMAN']].append(words)
            elif 'LOC' in category:
                test[cat2id['LOCATION']].append(words)
            elif 'NUM' in category:
                test[cat2id['NUMERIC']].append(words)

    with open(data_path + test_file, 'w') as json_file:
        json.dump(test, json_file, indent=2)


def create_vocabulary(embedding_model, data_path=data_path, train_file=train_file, val_file=val_file):
    """Create the vocabulary according to the train and val datasets.

    Args:
        embedding_model (gensim model): model attribute of the Embeddings class.
        data_path (str, optional): path to the dataset folder. Defaults to data_path.
        train_file (str, optional): path to train.json. Defaults to train_file.
        val_file (str, optional): path to val.json. Defaults to val_file.

    Returns:
        OrderedDict, OrderedDict: vocabularies to look up the index of the tokens and to look up the tokens according to the index.
    """
    stemmer = PorterStemmer()
    files = [train_file, val_file]
    data_dicts = []
    for file_ in files:
        with open(data_path + file_) as json_file:
            data_dicts.append(json.load(json_file))

    tokens = set()

    for data_dict in data_dicts:
        for sentences in data_dict.values():
            for sentence in sentences:
                tokens.update(map(stemmer.stem, sentence))
    token2ind = OrderedDict()
    ind2token = OrderedDict()
    token2ind['<unk>'] = 0
    token2ind['<pad>'] = 1
    ind2token[0] = '<unk>'
    ind2token[1] = '<pad>'
    i = 2
    for token in sorted(list(tokens)):
        if token not in embedding_model.vocab:
            token2ind[token] = 0
            continue
        token2ind[token] = i
        ind2token[i] = token
        i += 1

    return token2ind, ind2token


class QCDataset(data.Dataset):
    """Create the dataset for Experimental Data for Question Classification (https://cogcomp.seas.upenn.edu/Data/QA/QC/).

    Example usage:
        1. Given the created vocabulary (token2ind, ind2token), first initialize and instance of the QCDataset `qc_dataset = QCDataset(token2ind, ind2token)`.
        2. Use torch.utils.data.DataLoader() to create the generator `DataLoader(qc_dataset, batch_size=batch_size, shuffle=False, collate_fn=qc_dataset.collate_fn)`

    """

    def __init__(self, token2ind, ind2token, data_path=data_path, split='train', classes=all_classes):
        self.token2ind = token2ind
        self.ind2token = ind2token
        self.data_path = data_path
        self.splits = {
            'train': train_file,
            'val': val_file,
            'test': test_file
        }
        self.split = split
        self.classes = [cat2id[class_] for class_ in classes]
        self.num_classes = len(all_classes)
        self.stemmer = PorterStemmer()

        with open(self.data_path + self.splits[self.split]) as json_file:
            self.data = json.load(json_file)

        self.len = 0
        for class_ in self.classes:
            self.len += len(self.data[str(class_)])

    def __getitem__(self, i):
        """Generator for the tensors of the input sentence (list of words) and its target class.
        """
        target_class = np.random.choice(self.classes)
        # input_sentence = np.random.choice(self.data[str(target_class)])
        input_sentence = self.data[str(target_class)][i]
        input_sentence = [self.token2ind.get(token, 0) for token in map(self.stemmer.stem, input_sentence)]
        # return torch.LongTensor(input_sentence), torch.LongTensor(target_class)
        return input_sentence, target_class

    def __len__(self):
        return self.len

    def pad(self, sentence, max_length, pad_ind=1):
        return sentence + [pad_ind] * (max_length - len(sentence))

    def collate_fn(self, batch):
        """Combine the tensors in a batch, including padding sentences to have the same length.
        """
        input_sentences = []
        target_classes = []

        max_length = max([len(b[0]) for b in batch])
        for b in batch:
            one_hot_vector_input = np.zeros(shape=(max_length, len(self.ind2token)))
            one_hot_vector_input[np.arange(max_length), np.array(self.pad(b[0], max_length), dtype=np.int)] = 1.
            input_sentences.append(torch.LongTensor(one_hot_vector_input))
            one_hot_vector_target = np.zeros(self.num_classes)
            one_hot_vector_target[b[1]] = 1.
            target_classes.append(torch.LongTensor(one_hot_vector_target))
        # Dimension of input_sentences: batch_size x max_length x len(vocabulary)
        input_sentences = torch.stack(input_sentences, dim=0)
        # Dimension of target_classes: batch_size x len(classes)
        target_classes = torch.stack(target_classes, dim=0)
        return input_sentences, target_classes
