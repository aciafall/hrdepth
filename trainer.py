from __future__ import absolute_import, division, print_function
from random import shuffle

import numpy as np
import os
import glob

import torch
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import combinations
from collections import OrderedDict

import dataset
import network
import loss
import logger

class Trainer:
    def __init__(self):
        self.log_dir = './log_dir'
        self.load_weights_path = './ckpt'
        self.model_name = 'HR-Depth'
        self.log_path = os.path.join(self.log_dir, self.model_name)

        self.logger = logger.log_creater(self.log_path)
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.img_height = 352
        self.img_width = 640
        self.ratio = 0.33
        self.lr = 0.0001

        self.batch_size = 1 # only can be 1 because of the loss
        self.num_epochs = 50
        self.num_workers = 4
        self.best_acc = 0.0

        self.freeze_encoder = True

        self.encoder = network.ResnetEncoder(18, False).to(self.device)
        self.decoder = network.HRDepthDecoder(self.encoder.num_ch_enc)

        self.compute_losses = loss.Loss(ratio=self.ratio)
        
        self.optimizer = optim.SGD(self.decoder.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40,45], gamma=0.1)

        if self.load_weights_path is not None:
            self.load_model()

        if self.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        print("Training model named:\n  ", self.model_name)
        print("Models and log files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        self.logger.info("lr:{}, alpha:{}, freeze:{}, {}".format(self.lr, self.ratio, self.freeze_encoder, self.load_weights_path))

        # data
        self.data_dir = 'dataset'
        self.dataset = dataset.NuSceneDataset
        data = glob.glob('dataset/Images/samples/*')
        train_length = int(0.7 * len(data))
        test_length = len(data) - train_length
        train_split_path, test_split_path = random_split(data, [train_length, test_length], torch.Generator().manual_seed(233))
        train_split = []
        test_split = []
        for split in list(train_split_path):
            train_split.append(split.split('\\')[-1].split('.')[0])
        for split in list(test_split_path):
            test_split.append(split.split('\\')[-1].split('.')[0])

        train_dataset = self.dataset(self.data_dir, train_split, transform=dataset.Resize(self.width, self.height))
        self.train_loader = DataLoader(
            train_dataset, self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

        test_dataset = self.dataset(self.data_dir, test_split, transform=dataset.Resize(self.width, self.height))
        self.test_loader = DataLoader(
            test_dataset, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

        print("There are {:d} training items and {:d} test items\n".format(
            len(train_dataset), len(test_dataset)))

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0

        for self.epoch in range(self.num_epochs):
            self.run_epoch()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.decoder.train()

        for batch_idx, item in enumerate(self.train_loader):
            image, boxes = item[0], item[1]
            image, boxes = image.to(self.device), boxes.to(self.device)

            # outputs = self.model(image)['inv_depths'][0]
            outputs = self.decoder(self.encoder(image))

            losses = self.compute_losses(self.inv2depth(outputs), boxes[0])

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()
            
            self.log_info(batch_idx, losses)

        self.lr_scheduler.step()
        # self.save_model()
        self.val()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.model.eval()
        Accs, Vars = [], []
        for batch_idx, item in enumerate(self.test_loader):
            image, boxes = item[0], item[1]
            image = image.to(self.device)
            with torch.no_grad():
                inv_depth_pred = self.model(image)['inv_depths'][0]
                depth_pred = self.inv2depth(inv_depth_pred).cpu().numpy()

                acc, var = self.compute_mat(depth_pred, boxes[0].numpy())
                Accs.append(acc)
                Vars.append(var)

        self.logger.info("mean acc:{}, mean var:{}".format(float('%.4f' % np.mean(Accs)), float('%.4f' % np.mean(Vars))))

        if np.mean(Accs) > self.best_acc:
            self.best_acc = np.mean(Accs)
            self.save_best_model()
        else:
            self.logger.info("best_acc: {}".format(float('%.4f' % self.best_acc)))


    def compute_mat(self, d_pred, bboxes):
        """
        Calculates acc and var.

        """
        d_pred = d_pred[0][0]
        delta, t = 0, len(bboxes)
        us, var = [], []
        for i in range(t):
            bbox = bboxes[i].astype(np.int)
            depth_box = d_pred[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            us.append(np.mean(depth_box))
            depth_box = (depth_box - np.min(depth_box)) / (np.max(depth_box) - np.min(depth_box))
            var.append(np.var(depth_box))

        print(us,var)

        for i, j in combinations(range(t), 2):
            if us[i] < us[j]:
                delta += 1
        acc = 2*delta / (t ** 2 - t)
        var = np.mean(var)

        return acc, var

    def log_info(self, batch_idx, losses):
        """Print a logging statement to the terminal and logfile
        """
        print_string = "epoch {:>3} | batch {:>6} " + \
                       " | loss: {:.5f} | loss_acc: {:.5f} | loss_std: {:.5f} "
        self.logger.info(print_string.format(self.epoch, batch_idx, losses["loss"].item(), losses["loss_acc"].item(),
                                  losses["loss_std"].item()))

    def save_model(self):
        """Save model weights to disk
        """
        save_path = os.path.join(self.log_path, "{}_{}.pth".format(self.model_name, self.epoch))
        print('save model {}'.format(save_path))
        torch.save(self.model, save_path)

    def save_best_model(self):
        """Save best model weights to disk
        """
        save_path = os.path.join(self.log_path, "best.pth")
        self.logger.info('save best model {}, acc = {}'.format(save_path,self.best_acc))
        torch.save(self.model, save_path)

    def load_model(self):
        """Load model from disk
        """
        assert os.path.exists(self.load_weights_path), \
            "Cannot find file {}".format(self.load_weights_path)

        print("Loading pretrained weights...")
        path = self.load_weights_path
        encoder_path = os.path.join(path, 'encoder.pth')
        decoder_path = os.path.join(path, 'depth.pth')
        encoder_dict = torch.load(encoder_path)
        self.width = encoder_dict['width']
        self.height = encoder_dict['height']
        self.encoder = network.ResnetEncoder(18, False)
        self.decoder = network.HRDepthDecoder(self.encoder.num_ch_enc)
        self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in self.encoder.state_dict()})
        self.decoder.load_state_dict(torch.load(decoder_path))

        print('###### Pretrained model loaded:')

    def inv2depth(self, inv_depth):
            return 1. / inv_depth.clamp(min=1e-6)

    def same_shape(self, shape1, shape2):
        if len(shape1) != len(shape2):
            return False
        for i in range(len(shape1)):
            if shape1[i] != shape2[i]:
                return False
        return True
