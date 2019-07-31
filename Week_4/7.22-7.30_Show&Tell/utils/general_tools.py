import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def set_seed(seed):
	'''
	Fix immediate seed to repeat experiment
	:param seed: An integer
	:return:
	'''
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


def get_train_transform():
	'''
	Transform the format of train images
	:return:
	'''
	train_transform = transforms.Compose([
		transforms.Resize([300, 300], Image.ANTIALIAS), # 调整大小为300x300
		transforms.RandomCrop(224), # 尺寸为224的随机裁剪
		transforms.RandomHorizontalFlip(), # 随机水平翻转
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), # 图像标准化
							 (0.229, 0.224, 0.225))])
	return train_transform


def get_val_trainsform():
	'''
	Transform the format of validation images
	:return:
	'''
	val_transform = transforms.Compose([
		transforms.Resize([224, 224], Image.ANTIALIAS), # 调整大小为224x224
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), # 图像标准化
							 (0.229, 0.224, 0.225))])
	return val_transform


def adjust_lr(optimizer, shrink_factor):
	"""
	Shrinks learning rate by a specified factor.
	:param optimizer: optimizer whose learning rate must be shrunk.
	:param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
	"""
	print(" *** DECAYING learning rate. ***")
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * shrink_factor
	print("New learning rate is %f" % (optimizer.param_groups[0]['lr'],))


def coco_metrics(json_file, generated_captions_json, epoch, sentences_show_path):
	'''

	:param refs: Generated sentences
	:param json: Caption path of validation set
	:return:
	'''
	coco = COCO(json_file)
	cocorefs = coco.loadRes(generated_captions_json)
	cocoEval = COCOEvalCap(coco, cocorefs)
	# cocoEval.params['image_id'] = cocorefs.getImgIds()
	cocoEval.evaluate()


	if not os.path.exists(sentences_show_path):
		os.mkdir(sentences_show_path)

	for eva in cocoEval.evalImgs:

		imgId = eva['image_id']
		annIds = cocorefs.getAnnIds(imgIds=imgId)
		anns = cocorefs.loadAnns(annIds)

		if epoch == 1:
			imgId_df = pd.DataFrame(
				columns=['epoch', 'generated_caption', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L',
						 'CIDEr'])
			imgId_df.to_csv(sentences_show_path + '{}.csv'.format(imgId), index=0)
		imgId_df = pd.read_csv(sentences_show_path + '{}.csv'.format(imgId))
		score_dict = {'epoch': epoch,
					  'generated_caption': anns[0]['caption'],
					  'Bleu_1': round(eva['Bleu_1'],3),
					  'Bleu_2': round(eva['Bleu_2'],3),
					  'Bleu_3': round(eva['Bleu_3'],3),
					  'Bleu_4': round(eva['Bleu_4'],3),
					  'METEOR': round(eva['METEOR'],3),
					  'ROUGE_L': round(eva['ROUGE_L'],3),
					  'CIDEr': round(eva['CIDEr'],3),
					  }
		imgId_df = imgId_df.append(score_dict, ignore_index=True)
		imgId_df.to_csv(sentences_show_path + '{}.csv'.format(imgId), index=0)
	return cocoEval.eval.items()
