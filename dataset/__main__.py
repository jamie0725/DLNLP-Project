import argparse
from dataset.utils import parse_dataset, create_vocabulary, QCDataset
from preprocessing.utils import Embeddings
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Dataset commands.')
    parser.add_argument('--parse', default=False, type=bool, help='Parse original dataset into json.')
    args = parser.parse_args()
    if args.parse:
        parse_dataset()

    embeddings = Embeddings()
    token2ind, ind2token = create_vocabulary(embeddings.model)
    qc_dataset = QCDataset(token2ind, ind2token)
    qc_dataloader = DataLoader(qc_dataset, batch_size=5, collate_fn=qc_dataset.collate_fn, drop_last=True, pin_memory=True)

    for i, (batch_inputs, batch_targets) in enumerate(qc_dataloader):
        print(batch_inputs.size())
        print(batch_targets)
        break

    embeddings_vector = embeddings.create_embeddings(token2ind)
    print(embeddings_vector.shape)
    del embeddings


if __name__ == '__main__':
    main()
