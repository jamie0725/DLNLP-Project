import argparse
from dataset.utils import parse_dataset, create_vocabulary, QCDataset
from preprocessing.utils import Embeddings
from torch.utils.data import DataLoader
import pickle
 

def main():
    """Example usage of the QCDataset and Embeddings classes.
    """
    parser = argparse.ArgumentParser(description='Dataset commands.')
    parser.add_argument('--parse', default=False, type=bool, help='Parse original dataset into json.')
    args = parser.parse_args()
    if args.parse:
        parse_dataset()
    # Create an instance of the Embeddings class to load Gensim's Word2Vec word embeddings.
    embeddings = Embeddings()
    # Create the vocabulary given the train and val datasets and Word2Vec word embeddings.
    token2ind, ind2token = create_vocabulary(embeddings.model)
    # Create a truncated version of the Word2Vec word embeddings vector.
    embeddings_vector = embeddings.create_embeddings(token2ind)
    print(embeddings_vector.shape)
    # Get rid of Gensim's Word2Vec word embedding to release the memory.
    del embeddings
    # Instantiate the QCDataset and DataLoader accordingly.
    qc_dataset = QCDataset(token2ind, ind2token)
    qc_dataloader = DataLoader(qc_dataset, batch_size=16, collate_fn=qc_dataset.collate_fn, drop_last=True, pin_memory=True)

    # Use the DataLoader instance as a generator to load the batches.
    for i, (batch_inputs, batch_targets) in enumerate(qc_dataloader):
        print(batch_inputs.size())
        print(batch_targets)
        break

    embeddings_vector = embeddings.create_embeddings(token2ind)
    print(embeddings_vector.shape)
    with open('dataset/ind2token','wb') as f:
        pickle.dump(ind2token,f) 
        f.close()
    with open('dataset/token2ind','wb') as f:
        pickle.dump(token2ind,f) 
        f.close()
    with open('embeddings_vector','wb') as f:
        pickle.dump(embeddings_vector,f) 
    del embeddings


if __name__ == '__main__':
    main()
