import argparse
import os
import pprint
import shutil
import sys
import cv2

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
from config import config
import datasets
#from lib.config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet23_slim.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--source',
                        help="Video path",
                        default='challenge.mp4')

    args, unknown = parser.parse_known_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    module = eval('models.'+config.MODEL.NAME)
    module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = 'pretrained_models/best_val_smaller.pth' #.path.join(final_output_dir, 'best.pth')      
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')      
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    device = torch.device('cuda:0')
    model.to(device).cuda()
    #model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    
    start = timeit.default_timer()
    cap = cv2.VideoCapture(args.source)
    sv_pred = True
    model.eval()

    while True:
      ret, image = cap.read()
      
      with torch.no_grad():
        image = cv2.resize(image, test_size, interpolation = cv2.INTER_AREA)
        w, h, ch = image.shape
        size = (w, h)
        pred = multi_scale_inference(config, model, image, scales=config.TEST.SCALE_LIST, flip=config.TEST.FLIP_TEST)

        if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
          pred = F.interpolate(pred, size[-2:], mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        if sv_pred:
          # mean=[0.485, 0.456, 0.406],
          #  std=[0.229, 0.224, 0.225]
          image = image.squeeze(0)
          image = image.numpy().transpose((1,2,0))
          image *= [0.229, 0.224, 0.225]
          image += [0.485, 0.456, 0.406]
          image *= 255.0
          image = image.astype(np.uint8)

          _, pred = torch.max(pred, dim=1)
          pred = pred.squeeze(0).cpu().numpy()
          img8_out = map16.visualize_result(image, pred)
        msg = 'MeanIoU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \ Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)
        end = timeit.default_timer()
        logger.info('Mins: %d' % np.int((end-start)/60))

    cap.release()


if __name__ == '__main__':
    main()
