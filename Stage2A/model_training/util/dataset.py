import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from . import transform
import cv2

class MyDataset(Dataset):
	'''
	Create a dataset specific to SceneNetDataSet that will
	be use easily by PyTorch.
	'''
	def __init__(self,
				image_path,
				label_path,
				dataset_type='train',
				augmentations=None,
				train_h=473,
				train_w=473,
				scale_min=0.5,
				scale_max=2.0,
				rotate_min=-10,
				rotate_max=10,
				ignore_label=255,
				mean=None,
				has_predict=False,
				std=None):
		'''
		Parameters:
				image_path: the path to the image dataset
				label_path: the path to the label dataset
		'''
		#self.s3 = boto3.resource('s3')
		#self.bucket = self.s3.Bucket(path)
		#self.files = [obj.key for obj in self.bucket.objects.all()]
		assert (dataset_type in ['train', 'val', 'test'])
		self.has_predict = has_predict
		self.type = dataset_type
		self.image_names = list(sorted(os.listdir(image_path)))
		self.label_names = list(sorted(os.listdir(label_path)))
		self.image_names = [os.path.join(image_path,filename) for filename in self.image_names]
		self.label_names = [os.path.join(label_path,filename) for filename in self.label_names]
		self.train_h = train_h
		self.train_w = train_w
		self.ignore_label = ignore_label
		self.scale_min = scale_min
		self.scale_max = scale_max
		self.rotate_min = rotate_min
		self.rotate_max = rotate_max

		assert len(self.image_names) == len(self.label_names) # Check if the images folder and labels folder has the same length
		
		if mean is not None and std is not None: # The dataset come with its mean and std
			self.mean = mean # [0.3369, 0.3169, 0.3025]
			self.std = std # [0.1641, 0.1601, 0.1609]
		else: # Take the mean and std of ImageNet 
			self.mean = [0.485, 0.456, 0.406]
			self.mean = [item * 255 for item in self.mean]
			self.std = [0.229, 0.224, 0.225]
			self.std = [item * 255 for item in self.std]
		self.train_transform = transform.Compose([
			transform.RandScale([self.scale_min, self.scale_max]),
			transform.RandRotate([self.rotate_min, self.rotate_max], padding=self.mean, ignore_label=self.ignore_label),
			transform.RandomGaussianBlur(),
			transform.RandomHorizontalFlip(),
			transform.Crop([self.train_h, self.train_w], crop_type='rand', padding=self.mean, ignore_label=self.ignore_label),
			transform.ToTensor(),
			transform.Normalize(mean=self.mean, std=self.std)])
		self.val_transform = transform.Compose([
			transform.Crop([self.train_h, self.train_w], crop_type='center', padding=self.mean, ignore_label=self.ignore_label),
			transform.ToTensor(),
			transform.Normalize(mean=self.mean, std=self.std)])
		self.test_transform = transform.Compose([transform.ToTensor()])
		self.augmentations = augmentations

		
	def __len__(self):
		return len(self.image_names)
		
	def __getitem__(self,idx):
		#img = Image.open(self.image_names[idx]).convert('RGB')
		#mask = Image.open(self.label_names[idx])
		image = cv2.imread(self.image_names[idx], cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = np.float32(image)
		if self.type == 'test' and not (self.has_predict):
			mask = None
		else:
			mask = cv2.imread(self.label_names[idx], cv2.IMREAD_GRAYSCALE)       
			if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
				raise (RuntimeError("Image & label shape mismatch: " + self.image_names[idx] + " " + self.label_names[idx] + "\n"))
		if self.type == 'train' : # Apply preprocessing
			image,mask = self.train_transform(image,mask)
		elif self.type == 'val':
			image,mask = self.val_transform(image,mask)
			#return image, mask, self
		elif self.type == 'test':
			if not (self.has_predict):
				image,_ = self.test_transform(image,image[:,:,0])
				return image, self.image_names[idx]
			else:
				image, mask = self.test_transform(image,mask)
				return image,mask,self.image_names[idx]
		else :
			raise (RuntimeError('Undefined dataset type, must be train, test or val'))
		if self.augmentations: # Apply some data augmentations
			image, mask = self.augmentations(image,mask)
			#img = self.augmentations(mask)
			
		return image,mask