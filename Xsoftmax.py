from __future__ import print_function, absolute_import

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from pytorch_metric_learning.losses import cosface_loss
from IPython import embed

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss



class XSoftmax(nn.Module):
	# This is a basic class for softmaxCE loss
	def __init__(
			self, s: float, margin: float,
			num_classes: int, in_features: int,
			use_gpu: bool=True, smooth: bool=False
	):
		super(XSoftmax, self).__init__()
		self.margin = margin
		self.s = s
		self.weight = nn.Parameter(
			torch.randn(num_classes, in_features).float(), requires_grad=True
		)
		self.numclasses = num_classes
		self.use_gpu = use_gpu
		if smooth:
			self.loss = nn.CrossEntropyLoss()
		else:
			self.loss = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=use_gpu)


	def forward(self, inputs: Tensor, targets: Tensor):
		raise NotImplementedError


	def _linear(self, inputs: Tensor, weight: Tensor, input_norm: bool=True):
		if input_norm:
			x = F.normalize(inputs, dim=1)
		else:
			x = inputs
		w = F.normalize(weight, dim=1).cuda()
		if self.use_gpu:
			w = w.cuda()
		logit = torch.matmul(x, w.t())

		if logit.size() == (inputs.size(0), self.numclasses):
			return logit
		else:
			raise ValueError("error of logits")



class AmSoftmax(XSoftmax):
	# additive margin softmax
	def forward(self, inputs: Tensor, targets: Tensor):
		logits = self._linear(inputs, self.weight)

		m = self.s * self.margin
		mask = torch.zeros(logits.cpu().size()).scatter_(1, targets.unsqueeze(1).data.cpu(), m)
		if self.use_gpu:
			mask = mask.cuda()

		scaled_logits = self.s * logits - mask
		loss = self.loss(scaled_logits, targets)
		return loss


class Normsoftmax(XSoftmax):
	# normFace
	def forward(self, inputs: Tensor, targets: Tensor):
		logits = self._linear(inputs, self.weight)
		scaled_logits = self.s * logits
		loss = self.loss(scaled_logits, targets)

		return loss


class ArcSoftmax(XSoftmax):
	# ArcFace
	def forward(self, inputs: Tensor, targets: Tensor):
		pass

	def _linear(self, inputs: Tensor, weight: Tensor, input_norm: bool = True):
		return super()._linear(inputs, weight, input_norm)


if __name__ == '__main__':
	x = torch.rand(32,64).cuda()
	label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,3,3,3,3,1,1,1,1,2,2,2,2]).cuda()
	amsoftmax = AmSoftmax(64.0, 0.3, 5, 64, True, False)
	loss = amsoftmax(x, label)
