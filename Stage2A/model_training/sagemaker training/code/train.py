import argparse
import json
import logging
import os, time
import sagemaker_containers
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model.pspnet import PSPNet
from util.dataset import MyDataset
from util.util import AverageMeter, poly_learning_rate, pytorch_f1, pytorch_iou, pytorch_accuracy, intersectionAndUnionGPU

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
# Get Tensorboard writer
writer = SummaryWriter('/opt/ml/output/tensorboard/')






def create_model(nb_classes=9):
	'''
	Load a PSPNet model with a backend resnet50
	'''
	model = PSPNet(classes=nb_classes)
	#modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
	#modules_new = [model.ppm, model.cls, model.aux]
	#print(model)
	return model

def main_process(args):
	is_distributed = len(args.hosts) > 1 and args.backend is not None
	#logger.debug("Distributed training - {}".format(is_distributed))
	use_cuda = args.num_gpus > 0
	return not (is_distributed and use_cuda)

def _get_train_data_loader(batch_size, train_size, training_dir, is_distributed, workers, **kwargs):
	'''
	Get the dataset for validation.

	Args:
		batch_size      : batch_size for training
		train_size		: image size for training
		testing_dir     : path to the validation dataset directory
		is_distributed  : if the training is distributed
		workers		    : numbers of workers for dataloader
	'''
	logger.info("Get train data loader")
	dataset = MyDataset(training_dir+'/image',training_dir+'/mask',dataset_type='train',train_h=train_size,train_w=train_size)
	train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
										shuffle=train_sampler is None,
										sampler=train_sampler, drop_last=True, **kwargs)


def _get_test_data_loader(test_batch_size, train_size, testing_dir, is_distributed, workers, **kwargs):
	'''
	Get the dataset for validation.

	Args:
		test_batch_size : batch_size for validation
		train_size 		: image size for validation (same as those for training)
		testing_dir     : path to the validation dataset directory
		is_distributed  : if the validation is distributed
		workers		    : numbers of workers for dataloader
	'''
	logger.info("Get test data loader")
	dataset = MyDataset(testing_dir+'/image',testing_dir+'/mask',dataset_type='val',train_h=train_size, train_w=train_size)
	test_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
	return torch.utils.data.DataLoader(dataset,
										batch_size=test_batch_size, sampler=test_sampler,
										shuffle=False, **kwargs)

def load_model_from_weights(args,model):
	'''
	Load the model with weights get from precedent training.
	'''
	if os.path.isfile(args.weight):
		if main_process(args):
			logger.info("=> loading weight '{}'".format(args.weight))
		checkpoint = torch.load(args.weight)
		model.load_state_dict(checkpoint['state_dict'])
		if main_process(args):
			logger.info("=> loaded weight '{}'".format(args.weight))
	else:
		if main_process(args):
			logger.info("=> no weight found at '{}'".format(args.weight))

def load_model_from_checkpoints(args,model,optimizer,device):
	'''
	Restart the training from the last chekpoints.
	'''
	if os.path.isfile(args.resume):
		if main_process(args):
			logger.info("=> loading checkpoint '{}'".format(args.resume))
		# checkpoint = torch.load(args.resume)
		checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		#print(checkpoint['optimizer'])

# 		for state in optimizer.state.values():
# 			for k, v in state.items():
# 				if torch.is_tensor(v):
# 					state[k] = v.cpu().numpy()
		if main_process(args):
			logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		return model,optimizer       
	else:
		if main_process(args):
			logger.info("=> no checkpoint found at '{}'".format(args.resume))


def main(args):
	'''
	'''
	# Get the parameters for setting environment
	is_distributed = len(args.hosts) > 1 and args.backend is not None
	logger.debug("Distributed training - {}".format(is_distributed))
	use_cuda = args.num_gpus > 0
	logger.debug("Number of gpus available - {}".format(args.num_gpus))
	kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
	device = torch.device("cuda" if use_cuda else "cpu")
	is_multi_distributed = is_distributed and use_cuda
	# Get writer
	#writer = SummaryWriter(args.save_path)
	#writer = SummaryWriter('/opt/ml/output/tensorboard/')

	if is_distributed:
		# Initialize the distributed environment.
		world_size = len(args.hosts)
		os.environ['WORLD_SIZE'] = str(world_size)
		host_rank = args.hosts.index(args.current_host)
		os.environ['RANK'] = str(host_rank)
		dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
		logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
			args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
			dist.get_rank(), args.num_gpus))

	# set the seed for generating random numbers
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed(args.seed)
	
	# Get the train and test dataset loaders
	train_loader = _get_train_data_loader(args.batch_size, args.train_size, args.train_dir, is_distributed, args.workers, **kwargs)
	logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
																len(train_loader.sampler), len(train_loader.dataset),
																100. * len(train_loader.sampler) / len(train_loader.dataset)
				))
	print(args.evaluate, type(args.evaluate))
	if args.evaluate:
		test_loader = _get_test_data_loader(args.test_batch_size, args.train_size, args.test_dir, is_distributed, args.workers, **kwargs)
		logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
														len(test_loader.sampler), len(test_loader.dataset),
														100. * len(test_loader.sampler) / len(test_loader.dataset)
					))
	# Create the model
	model = create_model(nb_classes=args.classes).to(device)
	modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
	modules_new = [model.ppm, model.cls, model.aux]
	# Create the parameters to update
	params_list = []
	for module in modules_ori:
		params_list.append(dict(params=module.parameters(), lr=args.lr))
	for module in modules_new:
		params_list.append(dict(params=module.parameters(), lr=args.lr * 10))
	

	if is_multi_distributed:
		# multi-machine multi-gpu case
		model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Compute batchnorm stats across all the devices
		model = torch.nn.parallel.DistributedDataParallel(model)
		logger.debug("DistributedDataParallel")
	else:
		# single-machine multi-gpu case or single-machine or multi-machine cpu case
		model = torch.nn.DataParallel(model)
		logger.debug("DataParallel")

	# Create optimizer and loss function
	optimizer = optim.SGD(params_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

	if (args.resume):
		model,optimizer = load_model_from_checkpoints(args,model,optimizer,device)
	for epoch in range(args.start_epoch, args.epochs):
		epoch_log = epoch +1

		# Train the model
		logger.debug("Begin training.........")
		loss_train, mIoU_train, mAcc_train, mF1_train, allAcc_train = train(args,train_loader, model, optimizer, epoch, device,is_multi_distributed)
		# Write in tensorboard
		if main_process(args):
			writer.add_scalar('loss_train', loss_train, epoch_log)
			writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
			writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
			writer.add_scalar('mF1_train', mF1_train, epoch_log)
			writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
		
		if args.evaluate:
			loss_val, mIoU_val, mAcc_val, mF1_val, allAcc_val = validate(args,test_loader, model, device, criterion,is_multi_distributed)
			# Write in Tensorboard
			if main_process(args):
				writer.add_scalar('loss_val', loss_val, epoch_log)
				writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
				writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
				writer.add_scalar('mF1_val', mF1_val, epoch_log)
				writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
				
		# Save the model
		if (epoch_log % 10 == 0) :
			save_model(model,optimizer,epoch_log,args.model_dir)


def train(args,train_loader,model,optimizer,epoch, device,is_multi_distributed):
	'''
	'''
	# Compute statistics
	batch_time = AverageMeter()
	data_time = AverageMeter()
	main_loss_meter = AverageMeter()
	aux_loss_meter = AverageMeter()
	loss_meter = AverageMeter()
	iou_meter = AverageMeter()
	f1_meter = AverageMeter()
	accuracy_meter = AverageMeter()
	intersection_meter = AverageMeter()
	union_meter = AverageMeter()
	target_meter = AverageMeter()

	#
	end = time.time()
	max_iter = args.epochs * len(train_loader)
	model.train()      
	for batch_idx, (data, mask) in enumerate(train_loader, 1):
		data_time.update(time.time() - end)
		# Put the data in the format fitting the device
		data, mask = data.to(device), mask.to(device)
		
		# Forward
		output, main_loss, aux_loss = model(data,mask)

		# Compute loss
		if not(is_multi_distributed):
			main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
		loss = main_loss + args.aux_weight * aux_loss
		
		# Update optimizer
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		n = data.size(0)
		if is_multi_distributed:
			main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
			count = mask.new_tensor([n], dtype=torch.long)
			dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
			n = count.item()
			main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n
		
		# Compute metrics (Miou,union..)
		#if is_multi_distributed:
		intersection, union, target = intersectionAndUnionGPU(output, mask, args.classes, args.ignore_label)
		#else:
			#intersection, union, target = intersectionAndUnion(output, target, args.classes, args.ignore_label)

		if is_multi_distributed:
			dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
		intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
		intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

		accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
		#accuracy = pytorch_accuracy(target,output)
		f1 = pytorch_f1(mask,output)
		iou = sum(intersection_meter.val) / (sum(union_meter.val) + 1e-10)

		accuracy_meter.update(accuracy), f1_meter.update(f1), iou_meter.update(iou)
		main_loss_meter.update(main_loss.item(), n)
		aux_loss_meter.update(aux_loss.item(), n)
		loss_meter.update(loss.item(), n)
		batch_time.update(time.time() - end)
		end = time.time()

		# Update the learning rate
		current_iter = epoch * len(train_loader) + batch_idx + 1
		current_lr = poly_learning_rate(args.lr, current_iter, max_iter, power=args.power)
		for index in range(0, args.index_split):
			optimizer.param_groups[index]['lr'] = current_lr
		for index in range(args.index_split, len(optimizer.param_groups)):
			optimizer.param_groups[index]['lr'] = current_lr * 10
		
		# Compute the remaining time
		remain_iter = max_iter - current_iter
		remain_time = remain_iter * batch_time.avg
		t_m, t_s = divmod(remain_time, 60)
		t_h, t_m = divmod(t_m, 60)
		remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

		# Print information and stats
		if batch_idx % args.log_interval == 0:
			logger.info('Epoch: [{}/{}][{}/{}] '
						'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
						'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
						'Remain {remain_time} '
						'MainLoss {main_loss_meter.val:.4f} '
						'AuxLoss {aux_loss_meter.val:.4f} '
						'Loss {loss_meter.val:.4f} '
						'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, batch_idx + 1, len(train_loader),
														batch_time=batch_time,
														data_time=data_time,
														remain_time=remain_time,
														main_loss_meter=main_loss_meter,
														aux_loss_meter=aux_loss_meter,
														loss_meter=loss_meter,
														accuracy=accuracy))
		
		# Write to tensorboard
		if main_process(args):
			writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
			writer.add_scalar('mIoU_train_batch', np.mean(iou), current_iter)
			writer.add_scalar('f1_train_batch', np.mean(f1), current_iter)
			writer.add_scalar('mAcc_train_batch', np.mean(accuracy), current_iter)
			#writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
			#writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
			writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

	iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
	accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
	mIoU = np.mean(iou_class)
	mAcc = np.mean(accuracy_class)
	allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
	#mIoU = np.mean(iou_meter.avg)
	#mAcc = np.mean(accuracy_meter.avg)
	mF1 = np.mean(f1_meter.avg)
	#allAcc = accuracy_meter.sum
	if main_process(args):
		logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, mF1, allAcc))
	return main_loss_meter.avg, mIoU, mF1, mAcc, allAcc
			
def validate(args, test_loader, model, device, criterion,is_multi_distributed):
	'''
	'''
	if main_process(args):
		logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
	
	# Compute stats
	batch_time = AverageMeter()
	data_time = AverageMeter()
	loss_meter = AverageMeter()
	intersection_meter = AverageMeter()
	union_meter = AverageMeter()
	target_meter = AverageMeter()
	iou_meter = AverageMeter()
	f1_meter = AverageMeter()
	accuracy_meter = AverageMeter()

	model.eval()
	end = time.time()
	with torch.no_grad():
		for i, (data, mask) in enumerate(test_loader):
			data, mask = data.to(device), mask.to(device)
			output = model(data)
			
			# Compute loss
			loss = criterion(output,mask)
			n = data.size(0)
			if is_multi_distributed:
				loss = loss * n  # not considering ignore pixels
				count = mask.new_tensor([n], dtype=torch.long)
				dist.all_reduce(loss), dist.all_reduce(count)
				n = count.item()
				loss = loss / n
			else:
				loss = torch.mean(loss)

			output = output.max(1)[1]

			# Compute metrics
			#if is_multi_distributed:
			intersection, union, target = intersectionAndUnionGPU(output, mask, args.classes, args.ignore_label)
			#else:
				#intersection, union, target = intersectionAndUnion(output, target, args.classes, args.ignore_label)

			if is_multi_distributed:
				dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
			intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
			intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

			#accuracy = pytorch_accuracy(target,output)
			f1 = pytorch_f1(mask,output)
			iou = sum(intersection_meter.val) / (sum(union_meter.val) + 1e-10)

			accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
			accuracy_meter.update(accuracy), f1_meter.update(f1), iou_meter.update(iou)
			loss_meter.update(loss.item(), data.size(0))
			batch_time.update(time.time() - end)
			end = time.time()
			if ((i + 1) % args.log_interval == 0) and main_process(args):
				logger.info('Test: [{}/{}] '
							'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
							'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
							'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
							'Accuracy {accuracy:.4f}.'.format(i + 1, len(test_loader),
														data_time=data_time,
														batch_time=batch_time,
														loss_meter=loss_meter,
														accuracy=accuracy))
		iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
		accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
		mIoU = np.mean(iou_class)
		mAcc = np.mean(accuracy_class)
		allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

		#mIoU = np.mean(iou_meter.avg)
		#mAcc = np.mean(accuracy_meter.avg)
		mF1 = np.mean(f1_meter.avg)
		#allAcc = accuracy_meter.sum
		if main_process(args):
			logger.info('Val result: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, mF1, allAcc))
			#for i in range(args.classes):
				#logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
			logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
		return loss_meter.avg, mIoU, mAcc, mF1, allAcc



def model_fn(model_dir):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = torch.nn.DataParallel(create_model())
	cudnn.benchmark = True
	if os.path.isfile(model_dir):
		logger.info("=> loading checkpoint '{}'".format(model_dir))
		checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage.cuda())
		model.load_state_dict(checkpoint['state_dict'], strict=False)
		logger.info("=> loaded checkpoint '{}'".format(model_dir))
	else:
		raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
	return model.to(device)


def save_model(model, optimizer, epoch_log, model_dir):
	logger.info("Saving the model.")
	filename = os.path.join(model_dir, 'epochs_'+str(epoch_log)+'.pth')
	logger.info('Saving checkpoint to: ' + filename)
	torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

	# recommended way from http://pytorch.org/docs/master/notes/serialization.html
	#torch.save(model.cpu().state_dict(), path)
	#torch.save(model.cpu(), path)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Data and model checkpoints directories
	parser.add_argument('--batch-size', type=int, default=16, metavar='N',
						help='input batch size for training (default: 16)')
	parser.add_argument('--train-size', type=int, default=225, metavar='N',
						help='train image size for training (default: 225)')
	parser.add_argument('--evaluate', default=False,action='store_false',
						help='validate the model on the validation set or not')
	parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
						help='input batch size for testing (default: 8)')
	parser.add_argument('--epochs', type=int, default=100, metavar='N',
						help='number of epochs to train (default: 100)')
	parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
						help='the epoch number to start training (default: 0)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						help='SGD momentum (default: 0.9)')
	parser.add_argument('--power', type=float, default=0.9, metavar='M',
						help='SGD power (default: 0.9)')
	parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M',
						help='SGD weight decay (default: 0.0001)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--aux-weight', type=float, default=0.4, metavar='M',
						help='weight decay for loss (default: 0.4)')	
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--workers', type=int, default=1, metavar='N',
						help='how many workers for data loaders')
	parser.add_argument('--classes', type=int, default=9, metavar='N',
						help='how many classes for the model')
	parser.add_argument('--ignore-label',type=int, default=255, metavar='N',
						help='the label to ignored')
	parser.add_argument('--index-split',type=int, default=5, metavar='N',
						help='the label to ignored')
	parser.add_argument('--resume',type=str, default=None, metavar='M',
						help='the model path to restart training')
	parser.add_argument('--backend', type=str, default='nccl',
						help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

	# Container environment
	parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
	parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
	parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
	parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
	parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TESTING'])
	parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

	main(parser.parse_args())
