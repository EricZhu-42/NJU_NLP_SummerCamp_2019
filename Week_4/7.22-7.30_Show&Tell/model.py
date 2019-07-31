import heapq

import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

	def __init__(self,hidden_dim,fine_tuning):
		super(Encoder, self).__init__()

		cnn = models.resnet152(pretrained=True)
		modules = list(cnn.children())[:-2]
		self.cnn = nn.Sequential(*modules)
		self.affine_1 = nn.Linear(512, hidden_dim)
		for p in self.cnn.parameters():
			p.requires_grad = False
		if fine_tuning == True:
			self.fine_tune(fine_tuning=fine_tuning)

	def forward(self, images):

		features = self.cnn(images)
		features = features.permute(0, 2, 3, 1)
		features = features.reshape(features.size(0), -1,512)
		features = self.affine_1(features)
		return features

	def fine_tune(self, fine_tuning=False):
		for c in list(self.cnn.children())[7:]:
			for p in c.parameters():
				p.requires_grad = fine_tuning

class Decoder(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab, vocab_size, max_seq_length):

		super(Decoder, self).__init__()
		self.vocab_size = vocab_size
		self.vocab = vocab
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstmcell = nn.LSTMCell(embedding_dim, hidden_dim)
		self.fc = nn.Linear(hidden_dim, vocab_size)
		self.max_seq_length = max_seq_length
		self.init_h = nn.Linear(512, hidden_dim)
		self.init_c = nn.Linear(512, hidden_dim)

	def forward(self, features, captions, lengths, state=None):

		batch_size = features.size(0)
		vocab_size = self.vocab_size
		embeddings = self.embedding(captions)
		predictions = torch.zeros(batch_size, max(lengths), vocab_size).to(device)
		h, c = self.init_hidden_state(features)

		for t in range(max(lengths)):
			batch_size_t = sum([l > t for l in lengths])
			h, c = self.lstmcell(embeddings[:batch_size_t, t, :],
								 (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t,hidden_dim)
			preds = self.fc(h)
			predictions[:batch_size_t, t, :] = preds

		return predictions

	def generate(self, features, state=None):

		sentence = []
		h, c = self.init_hidden_state(features)
		input = self.embedding(torch.tensor([1]).to(device))

		for i in range(self.max_seq_length):

			h, c = self.lstmcell(input, (h, c))
			prediction = self.fc(h)
			_, prediction = prediction.max(1)
			word = self.vocab.idx2word[int(prediction)]
			if word == '<end>':
				break
			sentence.append(word)
			input = self.embedding(prediction)

		return sentence

	def test_generate(self, features, state=None):
		beam_size = 20
		h, c = self.init_hidden_state(features)
		old_list = [
			(list(),h,c,0.0,torch.tensor([1]))
		] * beam_size
		new_list = [
			(list(),h,c,0.0,torch.tensor([1]))
		] * beam_size

		for length in range(self.max_seq_length):
			for k in range(beam_size):
				old_data = old_list[k]
				if self.vocab.idx2word[old_data[4].cpu().numpy()[0]]=='<end>':
					continue
				input = self.embedding(old_data[4].to(device))
				h, c = self.lstmcell(input, (old_data[1],old_data[2]))
				predictions = self.fc(h).to(device)
				predictions = predictions.cpu().detach().numpy()
				idxs = heapq.nlargest(beam_size,range(predictions.size),predictions.take)
				for i in idxs:
					prob = predictions[0][i]
					new_prob = old_data[3] + prob
					if new_prob<new_list[-1][3] or new_prob==new_list[0][3]:
						break
					else:
						new_sentence = old_data[0] + [self.vocab.idx2word[int(i)]]
						new_list[-1] = (new_sentence,h,c,new_prob,torch.tensor([i]))
						new_list.sort(key=lambda x:x[3],reverse=True)
			old_list = new_list.copy()

		return new_list[0][0]

	def init_hidden_state(self, features):
		mean_features = features.mean(dim=1)
		h = self.init_h(mean_features)
		c = self.init_c(mean_features)
		return h, c

