import os
import pickle

import nltk
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from process import Vocabulary
from pycocotools.coco import COCO


class CocoTrainset(data.Dataset):

	def __init__(self, root, json, vocab, transform=None):
		'''

		:param root: Images path
		:param json: Captions path
		:param vocab: Vocabulary
		:param transform: Process images
		:return:
		'''

		self.root = root
		self.coco = COCO(json)
		self.ids = list(self.coco.anns.keys())
		self.vocab = vocab
		self.transform = transform # 对图像进行的变换处理

	def __getitem__(self, index):
		'''

		return a item
		'''
		coco = self.coco
		vocab = self.vocab
		ann_id = self.ids[index]

		caption = coco.anns[ann_id]['caption']
		img_id = coco.anns[ann_id]['image_id']
		path = coco.loadImgs(img_id)[0]['file_name']

		image = Image.open(os.path.join(self.root, path)).convert('RGB') # 读取图像
		if self.transform is not None:
			image = self.transform(image) # 应用变换

		tokens = nltk.tokenize.word_tokenize(str(caption).lower())
		caption = [] # 构造caption句子（列表）
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)
		return image, target, img_id # 返回图像，caption，id

	def __len__(self):
		return len(self.ids)


def train_collate_fn(data): # 训练集的样本获取函数
	'''

	:param data: -format:(image,caption,img_id)
	:return: images: tensor (batch_size,3,224,224)
			 targets: tensor (batch_size,padded_length)
			 lenghts: list,Every effective length of padding caption
			 img_ids:list,id of image
	'''
	# Sort by captions' length
	data.sort(key=lambda x: len(x[1]), reverse=True) # 按照caption的长度由小到大排序

	images, captions, img_ids = zip(*data)

	images = torch.stack(images, 0) # 将images堆叠

	lengths = [len(cap) for cap in captions]

	targets = torch.zeros(len(captions), max(lengths)).long() # 将图片存放至Tensor中

	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]

	return images, targets, lengths, list(img_ids)


def train_load(root, json, vocab, transform, batch_size, shuffle, num_workers):
	coco = CocoTrainset(root=root, json=json, vocab=vocab, transform=transform)

	data_loader = data.DataLoader(dataset=coco,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers,
								  collate_fn=train_collate_fn,
								  drop_last=True) # 抛弃长于BatchSize的数据

	return data_loader




class CocoValset(data.Dataset):

	def __init__(self, root, json, transform=None):
		'''

		:param root: Images path
		:param json: Captions path
		:param vocab: Vocabulary
		:param transform: Process images
		:return:
		'''

		self.root = root
		self.coco = COCO(json)
		self.ids = list(self.coco.imgs.keys())
		self.transform = transform

	def __getitem__(self, index): # 测试集取BatchSize==1
		'''
		return a item
		'''
		coco = self.coco
		img_id = self.ids[index]

		path = coco.loadImgs(img_id)[0]['file_name']

		image = Image.open(os.path.join(self.root, path)).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)

		return image, img_id

	def __len__(self):
		return len(self.ids)


def val_load(root, json, transform, batch_size, shuffle, num_workers):
	coco = CocoValset(root=root, json=json, transform=transform)

	data_loader = data.DataLoader(dataset=coco,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers,
								  )

	return data_loader
