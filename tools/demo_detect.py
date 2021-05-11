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
from torch.nn import functional as F

import _init_paths
import models
from config import config
import datasets
from datasets.base_dataset import BaseDataset
#from lib.config import config
from config import update_config
#from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test, Map16

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
    parser.add_argument('--show',
                        default=False,
                        type=bool,
                        help='If True display video')

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
        model_state_file = 'pretrained_models/best_val_smaller.pth'
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

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    
    cap = cv2.VideoCapture(args.source)
    sv_pred = True
    model.eval()
    print(config.DATASET.DATASET)
    bdataset = BaseDataset(num_classes=config.DATASET.NUM_CLASSES,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)
    
    map16 = Map16()

    while True:
      ret, image = cap.read()
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
      with torch.no_grad():
        h, w, _ = image.shape
        size = (h, w)
        image = bdataset.input_transform(image)
        image = image.transpose((2, 0, 1))
        pred = bdataset.multi_scale_inference(config, model, image)

        start = time.time()
        if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
            pred = F.interpolate(
                pred, size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

        if args.show:
          # mean=[0.485, 0.456, 0.406],
          #  std=[0.229, 0.224, 0.225]
          #image = image.squeeze(0)
          image = image.transpose((1,2,0))
          image *= [0.229, 0.224, 0.225]
          image += [0.485, 0.456, 0.406]
          image *= 255.0
          image = image.astype(np.uint8)

          _, pred = torch.max(pred, dim=1)
          pred = pred.squeeze(0).cpu().numpy()
          img8_out = map16.visualize_result(image, pred)
          cv2.imshow("DDRNET", img8_out)
          if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
      end = time.time()
      logger.info('Inference Performance in Seconds: %f, FPS: %f' % ((end-start) * 1000, (1/((end-start)*1000))))

    cap.release()


if __name__ == '__main__':
    main()
