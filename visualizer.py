import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import os
import numpy as np
import torch
import math
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Visualizer(object):
    """docstring for Visualizer"""
    def __init__(self):
        super(Visualizer, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.vis_saved_dir = os.path.join(self.opt.ckpt_dir, 'vis_pics')
        if not os.path.isdir(self.vis_saved_dir):
            os.makedirs(self.vis_saved_dir)
        plt.switch_backend('agg')
        self.plt_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        self.display_id = self.opt.visdom_display_id
        
   

    def print_losses_info(self, info_dict):
        msg = '[{}][Epoch: {:0>3}/{:0>3}; Images: {:0>4}/{:0>4}; Time: {:.3f}s/Batch({}); LR: {:.7f}] '.format(
                self.opt.name, info_dict['epoch'], info_dict['epoch_len'], 
                info_dict['epoch_steps'], info_dict['epoch_steps_len'], 
                info_dict['step_time'], self.opt.batch_size, info_dict['cur_lr'])
        for k, v in info_dict['losses'].items():
            msg += '| {}: {:.4f} '.format(k, v)
        msg += '|'
        print(msg)
        with open(info_dict['log_path'], 'a+') as f:
            f.write(msg + '\n')


    def tensor2im(self, input_image, imtype=np.uint8):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        return self.numpy2im(image_numpy, imtype)
        
    def numpy2im(self, image_numpy, imtype=np.uint8):
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))  
        # input should be [0, 1]
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        # print(image_numpy.shape)
        image_numpy = image_numpy.astype(imtype)
        im = Image.fromarray(image_numpy).resize((64, 64), Image.ANTIALIAS)
        return np.array(im)





