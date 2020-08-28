import os
import time
import logging
import argparse
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from util import dataset, transform
from config import config
from model.pspnet import PSPNet
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

cv2.ocl.setUseOpenCL(False)


def get_parser():
    '''
    Get the parser and arguments from the configuration file.
    '''
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/params_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='config/params_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    #parser.add_argument('--data_path',type=str,default='data/test/')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_logger():
    '''
    Create the logger to print message during the test.
    '''
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    '''
    Check if the arguments are correct.
    Args:
        args: arguments from command line.
    '''
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    assert args.metric in ['mIoU', 'accuracy']
    assert args.condition in ['inf', 'sup']
    if args.arch == 'psp':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    global args, logger
    args = get_parser()
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))


    # Get the dataset for evaluation or testing
    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    if not args.has_prediction:
        test_data = dataset.MyDataset(args.data_path,args.data_path,dataset_type='test')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        test_data = dataset.MyDataset(args.val_path+'image/', args.val_path+'mask/', dataset_type='test', has_predict=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Get mean and std of the dataset
    mean = test_data.mean
    std = test_data.std

    # Get the colors and the names of the classes of the dataset
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]

    # Create model and launch training
    if args.arch == 'psp':
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
    else:
        raise RuntimeError("this model is unknown or currently not supported !".format(args.arch))
    #logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    if not args.has_prediction:
        test(test_loader, model, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, gray_folder, color_folder, colors, names)
    else:
        val(test_loader, model, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, gray_folder, color_folder, colors, names)


def net_process(model, image, mean, std=None, flip=True):
    '''
    Pass an image to the model.
    Args:
        model: the model
        image: the image to pass
        mean: the mean of the dataset
        std: the standard deviation of the dataset
        flip:
    '''
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(test_loader, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors,names):
    '''
    Test the model on a dataset of images.
    Args:
        test_loader: The dataset test loader
        model: the model
        classes: the number of classes
        mean: the mean of the dataset (ImageNet mean)
        std: the standard deviation of the dataset (ImageNet std)
        base_size: the based size for scaling
        crop_h: the height for cropping
        crop_w: the width for cropping
        scales: the list of scales
        gray_folder: the folder to save gray labels
        color_folder: the folder to save color labels
        colors: the colors (RGB values) to apply to the gray labels
        names: the names of the classes
    '''
    logger.info('>>>>>>>>>>>>>>>> Start Test >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, image_path) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        
        # Get the image and its prediction
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image = Image.open(image_path[0],'r')
        #image_path, _ = data_list[i]
        image_name = image_path[0].split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')

        # Save the image and its prediction
        # Make an example plot with two subplots...
        if args.split == 'test':
            cv2.imwrite(gray_path, gray)
            fig = plt.figure()
            fig_im = fig.add_subplot(1,2,1)
            fig_im.imshow(np.asarray(image))
            fig_im.title.set_text('Image')

            fig_pred = fig.add_subplot(1,2,2)
            fig_pred.imshow(color)
            fig_pred.title.set_text('Prediction')

            # Make the legend for the colors and their class
            legend_patch = [mpatches.Patch(color=v/255,label=k) for k,v in zip(names,colors)]
            fig.legend(handles=legend_patch,loc="lower left", ncol=len(colors)//2)

            fig.savefig(color_path)

    logger.info('<<<<<<<<<<<<<<<<< End Test <<<<<<<<<<<<<<<<<')



def val(test_loader, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors, names):
    '''
    Evaluate the model on a dataset of images and their labels
    Args:
        test_loader: The dataset test loader
        model: the model
        classes: the number of classes
        mean: the mean of the dataset (ImageNet mean)
        std: the standard deviation of the dataset (ImageNet std)
        base_size: the based size for scaling
        crop_h: the height for cropping
        crop_w: the width for cropping
        scales: the list of scales
        gray_folder: the folder to save gray labels
        color_folder: the folder to save color labels
        colors: the colors (RGB values) to apply to the gray labels
        names: the names of each class.
    '''
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.eval()
    end = time.time()
    nb_img_ill = 0  # Image with poor scores

    # Create the csv file to save the model
    with open(args.score_file_path,'w',newline='') as f:
        writer = csv.writer(f,quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
        writer.writerow(['Image', 'Accuracy', 'mIoU'])

    for i, (input, target, image_path) in enumerate(test_loader):
        target = np.squeeze(target.numpy(),axis=0)
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        #check_makedirs(gray_folder)
        check_makedirs(color_folder)
        
        # Get the image and its prediction
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        gray_exp = np.uint8(target)
        expect = colorize(gray_exp,colors)
        image = Image.open(image_path[0],'r')
        #image_path, _ = data_list[i]
        image_name = image_path[0].split('/')[-1].split('.')[0]
        #gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')

        # Save the image and its prediction
        # Make an example plot with two subplots...
        #if args.split == 'val':
        # Compute statistics

        intersection, union, target = intersectionAndUnion(prediction, target, args.classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        iou = sum(intersection_meter.val) / (sum(union_meter.val) + 1e-10)
        #logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}, iou {4}.'.format(i + 1, len(data_list), image_name+'.png', accuracy, iou))

        # Save the results
        if args.metric == 'accuracy':
            score = accuracy
        elif args.metric == 'mIoU':
            score = iou
        else:
            raise RuntimeError('Unknown metric'.format(args.metric))
        # Put the score in a file with the name of the image
        with open(args.score_file_path,'a',newline='') as f:
            writer = csv.writer(f,quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
            writer.writerow([image_name,round(accuracy*100,3),round(iou*100,3)])
            #writer.writerow(f"Score of {round(score*100,3)}% for image {im_path} \n")
            #f.write("\n")
        
        if (args.condition == 'inf'):
            condition = (score <= args.threshold)
        else:
            condition = (score >= args.threshold)       
        if condition:
            nb_img_ill += 1

            # Save the results that satisfy the condition
            fig = plt.figure()
            fig_im = fig.add_subplot(1,3,1)
            fig_im.imshow(np.asarray(image))
            fig_im.title.set_text('Image')

            fig_exp = fig.add_subplot(1,3,2)
            fig_exp.imshow(expect)
            fig_exp.title.set_text('Expected')

            fig_pred = fig.add_subplot(1,3,3)
            fig_pred.imshow(color)
            fig_pred.title.set_text('Prediction')

            title = f"Score with {args.metric} metric of {round(score*100,3)}%"
            fig.suptitle(title)
            # Make the legend for the colors and their class
            legend_patch = [mpatches.Patch(color=v/255,label=k) for k,v in zip(names,colors)]
            fig.legend(handles=legend_patch,loc="lower left", ncol=len(colors)//2)
            fig.savefig(color_path)

    
    #if args.split == 'val':
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    

    with open(args.score_file_path,'a',newline='') as f:
        writer = csv.writer(f,quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
        writer.writerow([f"Average score with {args.metric} ",round(allAcc*100,3), round(mIoU*100,3) ])
        #writer.writerow('-'*50)
        #f.write("\n")
        #f.write(f"Score of {round(overall_score*100,3)}% for the all dataset \n")
        #f.write("\n")
    #logger.info("Test completed")
    logger.info("Total of images processed : {}".format(len(test_loader)))
    logger.info("The number of images with score {} to threshold is : {}".format(args.condition, nb_img_ill))
    logger.info("These images were saved in {}".format(args.save_folder))
    logger.info("Average score on the all dataset: {}%".format(round(allAcc*100,3)))

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))


    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
