import os
import numpy as np
from gensim.models import KeyedVectors


class Embeddings(object):
    """Class for creating an embedding vector with the pretrained Word2Vec embeddings.

    Example usage:
        Use torch.nn.Embed layer (with embedding size of 300) and fix the pretrained embeddings by
        `
        with torch.no_grad():
            model.embed.weight.data.copy_(torch.from_numpy(vectors))
            model.embed.weight.requires_grad = False
        `

    """

    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(os.path.dirname(os.path.realpath(__file__)) + '/GoogleNews-vectors-negative300.bin', binary=True)

    def create_embeddings(self, token2ind):
        vectors = []
        unknown_vector = np.random.uniform(-0.05, 0.05, 300).astype(np.float32)
        for token, ind in token2ind.items():
            if token == '<unk>':
                vectors.append(list(unknown_vector))
            elif token == '<pad>':
                vectors.append(list(np.zeros(300).astype(np.float32)))
            elif ind == 0:
                continue
            else:
                vectors.append(list(self.model[token]))

        vectors = np.stack(vectors, axis=0)
        return vectors
