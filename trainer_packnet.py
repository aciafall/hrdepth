from __future__ import absolute_import, division, print_function
from random import shuffle

import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import combinations
from collections import OrderedDict
from torch.utils.data import random_split
import glob

import dataset
# from evaluator import Evaluator
import network
import loss
import logger
import packnet


class Trainer:
    def __init__(self):
        self.log_dir = './log_dir'
        self.model_name = 'packnet'
        self.log_path = os.path.join(self.log_dir, self.model_name + '/newVarFormula')

        self.logger = logger.log_creater(self.log_path)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_height = 384
        self.img_width = 640
        # self.ratio = 0.33
        self.ratio = 0.33
        # self.lr = 0.0001
        self.lr = 0.0001

        self.batch_size = 1  # only can be 1 because of the loss
        self.num_epochs = 100
        self.num_workers = 6
        self.best_acc = 0.0

        self.model = packnet.PackNet01(dropout=None, version='1A').to(self.device)

        self.compute_losses = loss.Loss(ratio=self.ratio)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 90], gamma=0.1)

        self.load_weights_path = 'PackNet01_MR_selfsup_D.ckpt'

        if self.load_weights_path is not None:
            self.load_model()
            # self.model = torch.load('log_dir/packnet01A/0001/best.pth')

        self.freeze_encoder = True
        if self.freeze_encoder:
            self.model.freeze_encoder()

        print("Training model named:\n  ", self.model_name)
        print("Models and log files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        self.logger.info(
            "lr:{}, alpha:{}, freeze:{}, {}".format(
                self.lr, self.ratio, self.freeze_encoder, self.load_weights_path
            )
        )

        self.logger.info('DDAD transfer to KITTI')

        # data
        self.data_dir = 'dataset'
        self.dataset = dataset.VOCDataset
        data = glob.glob('dataset/VOC/JPEGImages/*')
        train_length = int(0.7 * len(data))
        test_length = len(data) - train_length
        train_split_path, test_split_path = random_split(data, [train_length, test_length],
                                                         torch.Generator().manual_seed(233))
        train_split = []
        test_split = []
        for split in list(train_split_path):
            train_split.append(split.split('\\')[-1].split('.')[0])
        for split in list(test_split_path):
            test_split.append(split.split('\\')[-1].split('.')[0])

        train_dataset = self.dataset(self.data_dir, train_split, transform=dataset.Resize())
        self.train_loader = DataLoader(
            train_dataset, self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

        test_dataset = self.dataset(self.data_dir, test_split, transform=dataset.Resize())
        self.test_loader = DataLoader(
            test_dataset, self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False)

        # train_dataset = self.dataset(self.data_dir, "train10_4", transform=dataset.Resize())
        # self.train_loader = DataLoader(
        #     train_dataset,
        #     self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        #     drop_last=False,
        # )
        #
        # test_dataset = self.dataset(self.data_dir, "test", transform=dataset.Resize())
        # self.test_loader = DataLoader(
        #     test_dataset,
        #     self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        #     drop_last=False,
        # )

        # self.evaluator = Evaluator(self.model, self.logger)

        print(
            "There are {:d} training items and {:d} test items\n".format(
                len(train_dataset), len(test_dataset)
            )
        )

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0

        for self.epoch in range(self.num_epochs):
            self.run_epoch()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # print("Training")
        self.model.train()

        for batch_idx, item in enumerate(self.train_loader):
            image, boxes = item[0], item[1]
            image, boxes = image.to(self.device), boxes.to(self.device)

            outputs = self.model(image)['inv_depths'][0]

            losses = self.compute_losses(self.inv2depth(outputs), boxes[0])

            losses["loss"].backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.log_info(batch_idx, losses)

        self.lr_scheduler.step()
        # self.save_model()
        self.val()
        # self.evaluator.evaluate()


    def val(self):
        """Validate the model on a single minibatch
        """
        self.model.eval()
        Accs, Vars, Stds = [], [], []
        for batch_idx, item in enumerate(self.test_loader):
            image, boxes = item[0], item[1]
            image = image.to(self.device)
            with torch.no_grad():
                inv_depth_pred = self.model(image)['inv_depths'][0]
                depth_pred = self.inv2depth(inv_depth_pred).cpu().numpy()

                acc, var, std = self.compute_mat(depth_pred, boxes[0].numpy())
                Accs.append(acc)
                Vars.append(var)
                Stds.append(std)

        self.logger.info(
            "mean acc:{}, mean var:{}, mean std:{}".format(
                float('%.4f' % np.mean(Accs)), float('%.4f' % np.mean(Vars)), float('%.4f' % np.mean(Stds))
            )
        )

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
        us, var, std = [], [], []
        for i in range(t):
            bbox = bboxes[i].astype(np.int)
            depth_box = d_pred[bbox[1]: bbox[3], bbox[0]: bbox[2]]

            ux = np.mean(depth_box)
            us.append(ux)
            # depth_box = (depth_box - np.min(depth_box)) / (np.max(depth_box) - np.min(depth_box))
            # ux = np.mean(depth_box)
            # var.append(np.mean(np.abs(depth_box - ux) / ux))
            var.append(np.var(depth_box / ux))
            # var.append(np.var(depth_box))
            std.append(np.std(depth_box / ux))

        # print(us,var)

        for i, j in combinations(range(t), 2):
            if us[i] < us[j]:
                delta += 1
        acc = 2 * delta / (t ** 2 - t)
        var = np.mean(var)
        std = np.mean(std)

        return acc, var, std

    def log_info(self, batch_idx, losses):
        """Print a logging statement to the terminal and logfile
        """
        print_string = "epoch {:>3} | batch {:>6} " + " | loss: {:.5f} | loss_acc: {:.5f} | loss_std: {:.5f} "
        self.logger.info(
            print_string.format(
                self.epoch,
                batch_idx,
                losses["loss"].item(),
                losses["loss_acc"].item(),
                losses["loss_std"].item(),
            )
        )

    def save_model(self):
        """Save model weights to disk
        """
        save_path = os.path.join(self.log_path, "{}_{}.pth".format(self.model_name, self.epoch))
        print('save model {}'.format(save_path))
        torch.save(self.model.state_dict(), save_path)

    def save_best_model(self):
        """Save best model weights to disk
        """
        save_path = os.path.join(self.log_path, "best.pth")
        self.logger.info('save best model {}, acc = {}'.format(save_path, self.best_acc))
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, prefixes=['depth_net', 'disp_network']):
        """Load model from disk
        """
        assert os.path.exists(self.load_weights_path), "Cannot find file {}".format(self.load_weights_path)

        print("Loading pretrained weights...")
        path = self.load_weights_path
        saved_state_dict = torch.load(path, map_location=self.device)['state_dict']

        network_state_dict = self.model.state_dict()

        updated_state_dict = OrderedDict()
        n, n_total = 0, len(network_state_dict.keys())
        for key, val in saved_state_dict.items():
            for prefix in prefixes:
                prefix = prefix + '.'
                if prefix in key:
                    idx = key.find(prefix) + len(prefix)
                    key = key[idx:]
                    if key in network_state_dict.keys() and self.same_shape(
                            val.shape, network_state_dict[key].shape
                    ):
                        updated_state_dict[key] = val
                        n += 1

        self.model.load_state_dict(updated_state_dict, strict=False)
        print('###### Pretrained {} loaded:'.format(prefixes[0]))

    def inv2depth(self, inv_depth):
        return 1.0 / inv_depth.clamp(min=1e-6)

    def same_shape(self, shape1, shape2):
        if len(shape1) != len(shape2):
            return False
        for i in range(len(shape1)):
            if shape1[i] != shape2[i]:
                return False
        return True

    def predict(self, input_path, output_path):
        pass
