# _*_ coding: utf-8 _*_

import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_length, embeddings_vector, lstm_layer, lstm_dirc, trainable, device):
        super(LSTMClassifier, self).__init__()

        self.embed = nn.Embedding.from_pretrained(embeddings_vector, freeze=trainable)
        self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers=lstm_layer, bidirectional=lstm_dirc, batch_first=True)
        if lstm_dirc:
            multiple = 2
        else:
            multiple = 1
        self.linear = nn.Linear(multiple * hidden_size, output_size)
        self.to(device)

    def forward(self, x):
        embed_input = self.embed(x)
        lstm_out, _ = self.lstm(embed_input)
        output = self.linear(lstm_out[:, -1, :].squeeze())

        return output
