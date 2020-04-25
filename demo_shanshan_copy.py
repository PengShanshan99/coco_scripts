from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchtext.vocab as vocab

import numpy as np
import time
import os
from six.moves import cPickle
import torch.backends.cudnn as cudnn
import yaml

import opts
import misc.eval_utils
import misc.utils as utils
import misc.AttModel as AttModel
import yaml

# from misc.rewards import get_self_critical_reward
import torchvision.transforms as transforms
import pdb
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
plt.switch_backend('agg')
import json
import pickle
import sys
sys.path.append('../faster-rcnn.pytorch/')
from get_ppls import prep_model, get_caption

def decode_sequence_simple(itow, itod, ltow, itoc, wtod, seq, bn_seq, fg_seq, vocab_size, opt):
    N, D = seq.size()
    itod = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            if j >= 1:
                txt = txt + ' '
            ix = seq[i,j].item()
            if ix > vocab_size:
                word = itod[int(fg_seq[i,j].item())]
                # word = '[ ' + word + ' ]'
            else:
                if ix == 0:
                    break
                else:
                    word = itow[str(ix)]
            txt = txt + word
        out.append(txt)
    return out

def demo_simple(obj_det_model, model, img_name):
    model.eval()
    file_path = img_name
    max_proposal = 200

    num_proposal = 6
    num_nms = 6

    # load the image.
    input_imgs = torch.FloatTensor(1)
    img = Image.open(file_path).convert('RGB')
    # resize the image.
    img = transforms.Resize((opt.image_crop_size, opt.image_crop_size))(img)
    
    ppls = get_caption(obj_det_model, img_name)
    pad_proposals = torch.cat([i.unsqueeze(dim=0) for i in ppls]).unsqueeze(dim=0)
#     pad_proposals = torch.from_numpy(pad_proposals).float().unsqueeze(dim=0)
    num = torch.FloatTensor([1, len(ppls), len(ppls)]).unsqueeze(dim=0)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img).unsqueeze(dim=0)
    
    img = img.cuda()
    pad_proposals = pad_proposals.cuda()
    num = num.cuda()

    eval_opt = {'sample_max':1, 'beam_size': opt.beam_size, 'inference_mode' : True, 'tag_size' : opt.cbs_tag_size}
    seq, bn_seq, fg_seq, _, _, _ = model._sample(img, pad_proposals, num, eval_opt)
    sents = utils.decode_sequence(dataset.itow, dataset.itod, dataset.ltow, dataset.itoc, dataset.wtod, \
                                    seq.data, bn_seq.data, fg_seq.data, opt.vocab_size, opt)
    return sents
    
if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        utils.update_values(options_yaml, vars(opt))
    print(opt)
    cudnn.benchmark = True

    ####################################################################################
    # Build the Model
    ####################################################################################
    from misc.dataloader_coco import DataLoader
    dataset = DataLoader(opt, split='train')
    opt.vocab_size = dataset.vocab_size
    opt.detect_size = dataset.detect_size
    opt.seq_length = opt.seq_length
    opt.fg_size = dataset.fg_size
    opt.fg_mask = torch.from_numpy(dataset.fg_mask).byte()
    opt.glove_fg = torch.from_numpy(dataset.glove_fg).float()
    opt.glove_clss = torch.from_numpy(dataset.glove_clss).float()
    opt.glove_w = torch.from_numpy(dataset.glove_w).float()
    opt.st2towidx = torch.from_numpy(dataset.st2towidx).long()

    opt.itow = dataset.itow
    opt.itod = dataset.itod
    opt.ltow = dataset.ltow
    opt.itoc = dataset.itoc

    if not opt.finetune_cnn: opt.fixed_block = 4 # if not finetune, fix all cnn block

    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    elif opt.att_model == 'att2in2':
        model = AttModel.Att2in2Model(opt)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        if opt.load_best_score == 1:
            model_path = os.path.join(opt.start_from, 'model-best.pth')
            info_path = os.path.join(opt.start_from, 'infos_'+opt.id+'-best.pkl')
        else:
            model_path = os.path.join(opt.start_from, 'model.pth')
            info_path = os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')

            # open old infos and check if models are compatible
        with open(info_path, 'rb') as f:
            infos = pickle.load(f, encoding='latin1')
            saved_model_opt = infos['opt']

        # opt.learning_rate = saved_model_opt.learning_rate
        print('Loading the model %s...' %(model_path))
        model.load_state_dict(torch.load(model_path))

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = pickle.load(f, encoding='latin1')

    if opt.decode_noc:
        model._reinit_word_weight(opt, dataset.ctoi, dataset.wtoi)

    best_val_score = infos.get('best_val_score', None)
    iteration = infos.get('iter', 0)
    start_epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    if opt.cuda:
        model.cuda()

    params = []
    # cnn_params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'cnn' in key:
                params += [{'params':[value], 'lr':opt.cnn_learning_rate,
                        'weight_decay':opt.cnn_weight_decay, 'betas':(opt.cnn_optim_alpha, opt.cnn_optim_beta)}]
            else:
                params += [{'params':[value], 'lr':opt.learning_rate,
                    'weight_decay':opt.weight_decay, 'betas':(opt.optim_alpha, opt.optim_beta)}]

    print("Use %s as optmization method" %(opt.optim))
    if opt.optim == 'sgd':
        optimizer = optim.SGD(params, momentum=0.9)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params)
    elif opt.optim == 'adamax':
    	optimizer = optim.Adamax(params)

#     img_name = 'test_images/COCO_val2014_000000184791.jpg'
    pics = ['COCO_val2014_000000184791.jpg','COCO_val2014_000000550529.jpg',
            'COCO_val2014_000000579664.jpg','COCO_val2014_000000348881.jpg','COCO_val2014_000000560623.jpg']
    obj_det_model = prep_model('../faster-rcnn.pytorch/data_fr/pretrained_model/faster_rcnn_coco.pth')
    for pic in pics:
        sents = demo_simple(obj_det_model, model, 'test_images/'+pic)
        print(sents)