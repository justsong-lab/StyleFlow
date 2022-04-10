import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import qdarkstyle
import qdarkgraystyle

from options.test_options import TestOptions
from ui.ui2 import Ui_Form

import numpy as np
from sklearn.neighbors import NearestNeighbors
from glob import glob
import cv2

from ui.mouse_event2 import GraphicsScene
from ui.GT_mouse_event import GTScene
from utils import Build_model
import pickle
from sklearn.manifold import TSNE
from ui.ui2 import transfer_real_to_slide, invert_slide_to_real, light_transfer_real_to_slide, \
    light_invert_slide_to_real, attr_degree_list
import torch
from module.flow import cnf
import os
from munch import Munch
import tensorflow as tf
from PIL import Image

from ui.real_time_attr_thread import RealTimeAttrThread
from ui.real_time_light_thread import RealTimeLightThread

# np.random.seed(2)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


self = Munch()
self.zero_padding = torch.zeros(1, 18, 1).cuda()
self.attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']


def main():
    latent_dict = np.load('results/latents.npy', allow_pickle=True).item()
    latent = list(latent_dict.values())[0][-1]
    latent = torch.tensor(latent).cuda().unsqueeze(0)
    opt = TestOptions().parse()
    init_deep_model(opt)
    # init_data_points()
    real_time_editing(latent, 0, 4)


def init_deep_model(opt):
    self.opt = opt
    self.model = Build_model(self.opt)
    self.w_avg = self.model.Gs.get_var('dlatent_avg')

    self.prior = cnf(512, '512-512-512-512-512', 17, 1)
    self.prior.load_state_dict(torch.load('flow_weight/modellarge10k.pt'))
    self.prior.eval()


def init_data_points():
    self.raw_w = pickle.load(open("data/sg2latents.pickle", "rb"))
    self.raw_TSNE = np.load('data/TSNE.npy')

    self.raw_attr = np.load('data/attributes.npy')
    self.raw_lights = np.load('data/light.npy')

    self.all_w = np.array(self.raw_w['Latent'])[self.keep_indexes]
    self.all_attr = self.raw_attr[self.keep_indexes]
    self.all_lights = self.raw_lights[self.keep_indexes]

    # self.X_samples = self.raw_TSNE[:self.sample_num]
    self.X_samples = self.raw_TSNE[self.keep_indexes]

    self.map = np.ones([1024, 1024, 3], np.uint8) * 255

    for point in self.X_samples:
        cv2.circle(self.map, tuple((point * 1024).astype(int)), 6, (0, 0, 255), -1)

    self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.X_samples)


def real_time_editing(latent, attr_index, raw_slide_value):
    # latent: [1, 18, 512]
    real_value = invert_slide_to_real(self.attr_order[attr_index], raw_slide_value)

    attr_change = real_value * self.pre_attr_distance[attr_index]
    attr_final = attr_degree_list[attr_index] * attr_change + self.attr_current_list[attr_index]
    self.q_array = torch.from_numpy(latent).cuda().clone().detach()

    self.final_array_target[0, attr_index + 9, 0, 0] = attr_final
    # the first arg is the latent
    # self.fws[0].shape
    # torch.Size([1, 18, 512])
    self.rev = self.prior(latent, self.final_array_target, self.zero_padding, True)

    if attr_index == 0:
        self.rev[0][0][8:] = self.q_array[0][8:]

    elif attr_index == 1:
        self.rev[0][0][:2] = self.q_array[0][:2]
        self.rev[0][0][4:] = self.q_array[0][4:]

    elif attr_index == 2:
        self.rev[0][0][4:] = self.q_array[0][4:]

    elif attr_index == 3:
        self.rev[0][0][4:] = self.q_array[0][4:]

    elif attr_index == 4:
        self.rev[0][0][6:] = self.q_array[0][6:]

    elif attr_index == 5:
        self.rev[0][0][:5] = self.q_array[0][:5]
        self.rev[0][0][10:] = self.q_array[0][10:]

    elif attr_index == 6:
        self.rev[0][0][0:4] = self.q_array[0][0:4]

        self.rev[0][0][8:] = self.q_array[0][8:]

    elif attr_index == 7:
        self.rev[0][0][:4] = self.q_array[0][:4]
        self.rev[0][0][6:] = self.q_array[0][6:]

    self.w_current = self.rev[0].detach().cpu().numpy()
    self.q_array = torch.from_numpy(self.w_current).cuda().clone().detach()

    self.fws = self.prior(self.q_array, self.final_array_target, self.zero_padding)
    GAN_image = self.model.generate_im_from_w_space(self.w_current)[0]
    img = Image.fromarray(GAN_image, 'RGB')
    img.save('tmp.png')


if __name__ == '__main__':
    main()
