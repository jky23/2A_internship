import torch
import numpy as np
from pytorch_lightning.metrics.functional import f1_score, iou, accuracy

class Metric():
	'''
	Class that implements metrics for image segmentation.
	'''
	
	def __init__(self,name,num_classes=9,custom_version=False):
		'''
			Args:
				name: the name of the metric that will be used
				num_classes: the number of classes in the segmentation model
				custom_version: if you will use the custom version or not
		'''
		self.name = name
		self.num_classes = num_classes
		self.classes = [i for i in range(self.num_classes)]
		self.custom_version = custom_version
		
	def call(self,y_true,y_pred):
		'''
		Call the metric which the name was passed as arguments.
			Args:
				y_true: the true label 
				y_pred: the predicted label		
		'''
		if (self.name).lower() == 'miou':
			return self.mIOU(y_true,y_pred) if self.custom_version else self.pytorch_iou(y_true,y_pred)
		if (self.name).lower() == 'f1_score' or (self.name).lower() =='dice':
			return self.f1_score(y_true,y_pred) if self.custom_version else self.pytorch_f1(y_true,y_pred)
		if (self.name).lower() == 'pixel_accuracy':
			return self.pixel_accuracy(y_true,y_pred) if self.custom_version else self.pytorch_accuracy(y_true,y_pred)
		
	def pytorch_iou(self,y_true,y_pred):
		'''
		Computes the pytorch IOU metrics.
			Args:
				y_true: the true label 
				y_pred: the predicted label	
		'''
		return iou(y_true,y_pred,num_classes=9).item()
		
	def pytorch_f1(self,y_true,y_pred):
		'''
		Computes the pytorch F1_score metric.
			Args:
				y_true: the true label 
				y_pred: the predicted label	
		'''
		return f1_score(y_true,y_pred).item()
		
	def pytorch_accuracy(self,y_true,y_pred):
		'''
		Computes the pytorch accuracy metric.
			Args:
				y_true: the true label 
				y_pred: the predicted label	
		'''
		return accuracy(y_true,y_pred).item()
	
	def mIOU(self,y_true, y_pred):
		'''
		Computes a custom version of IOU metrics.
			Args:
				y_true: the true label 
				y_pred: the predicted label	
		'''
		iou_list = list()
		present_iou_list = list()

		y_pred = y_pred.view(-1)
		print(y_pred.shape)
		y_true = y_true.view(-1)
		print(y_true.shape)
		# Note: Following for loop goes from 0 to (num_classes-1)
		# and ignore_index is num_classes, thus ignore_index is
		# not considered in computation of IoU.
		for sem_class in range(self.num_classes):
			y_pred_inds = (y_pred == sem_class)
			target_inds = (y_true == sem_class)
			if target_inds.long().sum().item() == 0:
				iou_now = float('nan')
			else: 
				intersection_now = (y_pred_inds[target_inds]).long().sum().item()
				union_now = y_pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
				iou_now = float(intersection_now) / float(union_now)
				present_iou_list.append(iou_now)
			iou_list.append(iou_now)
		return np.mean(present_iou_list)
		
		
	def pixel_accuracy(self,y_true,y_pred):
		'''
		Computes a custom version of accuracy metrics.
			Args:
				y_true: the true label 
				y_pred: the predicted label	
		'''
		total_pixels = y_true.nelement()
		correct_pixels = y_pred.eq(y_true).sum().item()
		return correct_pixels/total_pixels




	def f1_score(self,y_true,y_pred):
		'''
		Computes a custom version of F1 Score metric.
		This version is not working well (DO NOT USE)
			Args:
				y_true: the true label 
				y_pred: the predicted label	
		'''
		if y_pred.ndim == 3:
			y_pred = y_pred.squeeze(0)
			
		# Reshape into 1D tensors
		y_pred = y_pred.view(-1)
		y_true = y_true.view(-1)
		
		tp = (y_true * y_pred).sum().to(torch.float32)
		tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
		fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
		fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
		
		epsilon = 1e-7
		
		precision = tp / (tp + fp + epsilon)
		recall = tp / (tp + fn + epsilon)
		
		f1 = 2* (precision*recall) / (precision + recall + epsilon)
		return f1.item()



'''
metric = Metric('Miou',num_classes=5,custom_version=True)
metric1 = Metric('Miou',num_classes=5,custom_version=False)
metric2 = Metric('dice',num_classes=5,custom_version=True)
metric3 = Metric('dice',num_classes=5,custom_version=False)
metric4 = Metric('pixel_accuracy',num_classes=5,custom_version=True)
metric5 = Metric('pixel_accuracy',num_classes=5,custom_version=False)
output = torch.randn(1,5,10, 10)
target = torch.randint(0, 5, (10,10))
preds = torch.argmax(output, 1)
preds.squeeze(0)
print(preds.shape)
#score = metric.iou(preds,target)
score = metric.call(target,preds)
score1 = metric1.call(target,preds)
score2 = metric2.call(target,preds)
score3 = metric3.call(target,preds)
score4 = metric4.call(target,preds)
score5 = metric5.call(target,preds)
print("MIOU",score)
print("MIOU sk",score1)
print("dice",score2)
print("dice sk",score3)
print("PA",score4)
print("PA sk",score5)
'''
