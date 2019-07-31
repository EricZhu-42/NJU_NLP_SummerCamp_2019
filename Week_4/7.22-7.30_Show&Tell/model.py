import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import pickle

# class Vocabulary(object):
#
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = {}
#         self.idx = 0
#
#     def add_word(self,word):
#         if not word in self.word2idx:
#             self.word2idx[word] = self.idx
#             self.idx2word[self.idx] = word
#             self.idx+=1
#
#     def __call__(self,word):
#
#         # If a word not in vocabulary,it will be replace by <unknown>
#         if not word in self.word2idx:
#             return self.word2idx['<unk>']
#         return self.word2idx[word]
#
#     def __len__(self):
#         return len(self.word2idx)

class Encoder(nn.Module): # 使用ResNet152并加以调整，图片->包含语义信息的向量

	def __init__(self,hidden_dim,fine_tuning):

		super(Encoder, self).__init__() # 调用nn.module的__init__方法
		cnn = models.resnet152(pretrained=True) # 使用经过预训练的ResNet152网络权重
		modules = list(cnn.children())[:-2] # 选择除了最后两层（全连接层）的网络层
		self.cnn = nn.Sequential(*modules) # 将层传入nn.Sequential中
		self.affine_1 = nn.Linear(512, hidden_dim) # 定义最后一层全连接层
		for p in self.cnn.parameters(): # 特征提取部分的网络无需训练
			p.requires_grad = False
		if fine_tuning == True: # 规定fine-tune
			self.fine_tune(fine_tuning=fine_tuning)

	def forward(self, images):

		features = self.cnn(images) # 提取特征
		features = features.permute(0, 2, 3, 1) # 维度重排
		features = features.reshape(features.size(0), -1,512) # 将输出值在保持BatchSize的前提下展平为512维向量
		features = self.affine_1(features) # 传入全连接层
		return features

	def fine_tune(self, fine_tuning=False): # 定义fine-tine，即对CNN部分进行优化
		for c in list(self.cnn.children())[7:]: # 只优化ResNet后面的部分，防止影响到基础特征抽取部分，造成结果的不稳定
			for p in c.parameters():
				p.requires_grad = fine_tuning


class Decoder(nn.Module): # 使用多个LSTM单元完成，语义向量->自然语言句子

	def __init__(self, embedding_dim, hidden_dim, vocab, vocab_size, max_seq_length):

		super(Decoder, self).__init__() # 基础定义部分
		self.vocab_size = vocab_size
		self.vocab = vocab
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstmcell = nn.LSTMCell(embedding_dim, hidden_dim) # 使用LSTM单元
		self.fc = nn.Linear(hidden_dim, vocab_size)
		self.max_seq_length = max_seq_length
		self.init_h = nn.Linear(512, hidden_dim) # 初始化 Hidden-layer
		self.init_c = nn.Linear(512, hidden_dim) # 初始化 Cell-state

	def forward(self, features, captions, lengths, state=None):

		batch_size = features.size(0) # 获取BatchSize
		vocab_size = self.vocab_size
		embeddings = self.embedding(captions) #创建对captions的embedding
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
		input = self.embedding(torch.tensor([1]).to(device)) # 首先输入<start>的索引

		for i in range(self.max_seq_length):

			h, c = self.lstmcell(input, (h, c))
			prediction = self.fc(h)
			_, prediction = prediction.max(1) # 获取概率最大值的prediction
			word = self.vocab.idx2word[int(prediction)] # 转换为字符
			if word == '<end>':
				break
			sentence.append(word)
			input = self.embedding(prediction)

		return sentence

	def init_hidden_state(self, features): # 用features算出h与c

		mean_features = features.mean(dim=1)
		h = self.init_h(mean_features)
		c = self.init_c(mean_features)
		return h, c
