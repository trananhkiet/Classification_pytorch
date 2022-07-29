import torch
import numpy as np
import os

from torch import nn
from torch.nn import CrossEntropyLoss

def accuracy(data_iter, model, device):
	count_true = total = 0
	error_0 = 0
	error_1 = 0

	for img, label in iter(data_iter):
		img = img.to(device)
		label = label.numpy()
		score = model(img).cpu().numpy()
		total += score.shape[0]
		for i in range(score.shape[0]):
			if(np.argmax(score[i]) == label[i]):
				count_true += 1
			else:
				if label[i] == 0:
					error_0 += 1
				else:
					error_1 += 1

	return count_true * 100 / total

def val_loss(data_iter, model, device):
	loss = total = 0

	for img, label in iter(data_iter):
		img = img.to(device)
		# print(img)
		label = label.to(device)
		# print(label)
		score = model(img)
		total += 1
		loss += CrossEntropyLoss()(score, label).cpu().numpy()

	return loss / total


def write_txt(path, step, val, acc, loss):
	path = path + '/model_' + str(step) + '/'
	os.makedirs(os.path.dirname(path), exist_ok=True)
	f = open(path + 'stat.txt', 'w')
	f.write('Loss: ' + str(loss.detach().cpu().numpy()) + '\n')
	f.write('Validation: ' + str(val) + '\n')
	f.write('Accuracy: ' + str(acc) + '%' + '\n')
	f.close()