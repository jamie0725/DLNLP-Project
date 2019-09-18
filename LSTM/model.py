# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length):
		super(LSTMClassifier, self).__init__()
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.embed = nn.Embedding(vocab_size, embedding_length,padding_idx=1)
		self.lstm = nn.LSTM(embedding_length, hidden_size)
		self.label = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		batch_size = x.shape[1]
		input = self.embed(x)
		 # Initial hidden state  the LSTM
		h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
		c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
		lstm_out, (h_n, c_n) = self.lstm(input, (h_0, c_0))
		output = self.label(h_n[-1])
		return output