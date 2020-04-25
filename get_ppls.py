# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import imutils
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from matplotlib.pyplot import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='../faster-rcnn.pytorch/cfgs_fr/res101_ls.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models')
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--path_opt', type=str) 
  parser.add_argument('--batch_size', type=int) 
  parser.add_argument('--num_workers', type=int)
  parser.add_argument('--max_epoch', type=int)
  parser.add_argument('--inference_only', type=bool)
  parser.add_argument('--beam_size', type=int)
  parser.add_argument('--start_from', type=str)
  parser.add_argument('--image_path', type=str)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def prep_model(input_dir):

  args = parse_args()

  print('Called with args:')
  print(args)
  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = 1

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir)

#   pascal_classes = np.asarray(['__background__',
#                        'aeroplane', 'bicycle', 'bird', 'boat',
#                        'bottle', 'bus', 'car', 'cat', 'chair',
#                        'cow', 'diningtable', 'dog', 'horse',
#                        'motorbike', 'person', 'pottedplant',
#                        'sheep', 'sofa', 'train', 'tvmonitor'])
  pascal_classes = np.asarray(['__background__',"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
                               "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
                               "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
                               "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
                               "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
                               "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
                               "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
                               "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
                               "microwave","oven","toaster","sink",
                               "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"])

  # initilize the network here.
  fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
#   fasterRCNN.load_state_dict(checkpoint)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  print('load model successfully!')
  return fasterRCNN

  # pdb.set_trace()
def get_caption(fasterRCNN, img_path):
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  im_data = im_data.cuda()
  im_info = im_info.cuda()
  num_boxes = num_boxes.cuda()
  gt_boxes = gt_boxes.cuda()

  # make variable
  with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

  cfg.CUDA = True

  fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = True

#   while (num_images >= 0):
  total_tic = time.time()

  # Load the demo image
  im_file = img_path
  # im = cv2.imread(im_file)
  im_in = np.array(imread(im_file))
  if len(im_in.shape) == 2:
    im_in = im_in[:,:,np.newaxis]
    im_in = np.concatenate((im_in,im_in,im_in), axis=2)
    # rgb -> bgr
  im_in = im_in[:,:,::-1]
  im = im_in

  blobs, im_scales = _get_image_blob(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"
  im_blob = blobs
  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  im_data_pt = torch.from_numpy(im_blob)
  im_data_pt = im_data_pt.permute(0, 3, 1, 2)
  im_info_pt = torch.from_numpy(im_info_np)

  im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
  im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
  gt_boxes.data.resize_(1, 1, 5).zero_()
  num_boxes.data.resize_(1).zero_()

  # pdb.set_trace()
  det_tic = time.time()

  rois, cls_prob, bbox_pred, \
  rpn_loss_cls, rpn_loss_box, \
  RCNN_loss_cls, RCNN_loss_bbox, \
  rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

  scores = cls_prob.data
  boxes = rois.data[:, :, 1:5]

  # Simply repeat the boxes, once for each class
  _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
  pred_boxes = _.cuda()

  pred_boxes /= im_scales[0]

  scores = scores.squeeze()
  pred_boxes = pred_boxes.squeeze()
  det_toc = time.time()
  detect_time = det_toc - det_tic
  misc_tic = time.time()
  if vis:
    im2show = np.copy(im)
  detections = []
#   pascal_classes = np.asarray(['__background__',
#                        'aeroplane', 'bicycle', 'bird', 'boat',
#                        'bottle', 'bus', 'car', 'cat', 'chair',
#                        'cow', 'diningtable', 'dog', 'horse',
#                        'motorbike', 'person', 'pottedplant',
#                        'sheep', 'sofa', 'train', 'tvmonitor'])
  pascal_classes = np.asarray(['__background__',"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
                               "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
                               "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
                               "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
                               "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
                               "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
                               "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
                               "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
                               "microwave","oven","toaster","sink",
                               "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"])
  for j in xrange(1, len(pascal_classes)):
    inds = torch.nonzero(scores[:,j]>thresh).view(-1)
    # if there is det
    if inds.numel() > 0:
      cls_scores = scores[:,j][inds]
      _, order = torch.sort(cls_scores, 0, True)
#       if args.class_agnostic:
      cls_boxes = pred_boxes[inds, :]
      cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
      # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
      cls_dets = cls_dets[order]
      keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
      cls_dets = cls_dets[keep.view(-1).long()]
      detections.append(torch.cat([cls_dets[0][:4],torch.FloatTensor([j]).cuda(), cls_dets[0][-1:]]))
  return detections


