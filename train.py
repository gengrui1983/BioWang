import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from path import Path
from tensorboardX import SummaryWriter

from dataset.dataset import BioData

import models.BioModelCnn
from logger import TermLogger, AverageMeter

parser = argparse.ArgumentParser(description='Bio data experiments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.evaluate:
        args.epochs = 0

    train_set = BioData(args.root, seed=args.seed, train=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(
    #     val_set, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    bio_net = models.BioModelCnn.to(device)
    bio_net.init_weights()

    cudnn.benchmark = True
    bio_net = torch.nn.DataParallel(bio_net)

    print('=> setting adam solver')

    optim_params = [
        {'params': bio_net.parameters(), 'lr': args.lr},
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, bio_net, optimizer, args.epoch_size, logger)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))
        logger.reset_valid_bar()

    logger.epoch_bar.finish()


def train(args, train_loader, bio_net, optimizer, epoch_size, logger):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight

    # switch to train mode
    bio_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data = sample.to(device)

        # compute output
        estimated_y = bio_net(data['X'])

        loss = data['y'] - estimated_y

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]

if __name__ == '__main__':
    main()
