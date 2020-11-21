import torch
import torch.nn.functional as F
import torch.nn as nn

def bpr_loss(positive_predictions, negative_predictions, mask=None):
	loss = (1.0 - F.sigmoid(positive_predictions - negative_predictions))
	if mask is not None:
		mask = mask.float()
		loss = loss * mask
		return loss.sum() / mask.sum()
	return loss.mean()

class ScaledEmbedding(nn.Embedding):
	def reset_parameters(self):
		self.weight.data.normal_(0, 1.0 / self.embedding_dim)
		if self.padding_idx is not None:
			self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
	def reset_parameters(self):
		self.weight.data.zero_()
		if self.padding_idx is not None:
			self.weight.data[self.padding_idx].fill_(0)