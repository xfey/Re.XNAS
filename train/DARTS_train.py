"""train indepandent (can be augmented) model for DARTS"""
"""only support cifar10 now"""

import os
import argparse
import numpy as np
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import xnas.core.config as config
import xnas.core.distributed as dist
import xnas.core.logging as logging
import xnas.core.meters as meters
import xnas.search_space.cellbased_basic_genotypes as gt

from xnas.core.config import cfg
from xnas.core.builders import build_loss_fun, lr_scheduler_builder
from xnas.core.trainer import setup_env, setup_model, test_epoch
from xnas.search_space.cellbased_DARTS_cnn import AugmentCNN
from xnas.datasets.loader import construct_loader

device = torch.device("cuda")

writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)

# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()


"""
--genotype "Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('skip_connect', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 3), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 4)],
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],
        [('sep_conv_3x3', 3), ('skip_connect', 2)],
        [('dil_conv_3x3', 3), ('sep_conv_5x5', 0)],
    ],
    reduce_concat=range(2, 6))"
"""

def main():
    setup_env()

    input_size, input_channels, n_classes, train_data, valid_data = get_data(
        cfg.SEARCH.DATASET, cfg.SEARCH.DATAPATH, cfg.TRAIN.CUTOUT_LENGTH, validation=True)

    loss_fun = build_loss_fun().cuda()
    use_aux = cfg.TRAIN.AUX_WEIGHT > 0.

    model = AugmentCNN(input_size, input_channels, cfg.TRAIN.INIT_CHANNELS, n_classes, cfg.TRAIN.LAYERS,
                       use_aux, cfg.TRAIN.GENOTYPE)
    
    # TODO: Parallel
    # model = nn.DataParallel(model, device_ids=cfg.NUM_GPUS).to(device)
    model.cuda()

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), cfg.OPTIM.BASE_LR, momentum=cfg.OPTIM.MOMENTUM,
                                weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    
    # Get data loader
    [train_loader, valid_loader] = construct_loader(
        cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)

    lr_scheduler = lr_scheduler_builder(optimizer)

    best_top1 = 0.

    # TODO: DALI backend support
    # if config.data_loader_type == 'DALI':
    #     len_train_loader = get_train_loader_len(config.dataset.lower(), config.batch_size, is_train=True)
    # else:
    len_train_loader = len(train_loader)
    
    # Training loop
    # TODO: RESUME
    
    train_meter = meters.TrainMeter(len(train_loader))
    valid_meter = meters.TestMeter(len(valid_loader))

    start_epoch = 0
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        
        drop_prob = cfg.TRAIN.DROP_PATH_PROB * cur_epoch / cfg.OPTIM.MAX_EPOCH
        if cfg.NUM_GPUS > 1:
            model.module.drop_path_prob(drop_prob)
        else:
            model.drop_path_prob(drop_prob)
    
        # Training 
        train_epoch(train_loader, model, optimizer, loss_fun, cur_epoch, train_meter)

        lr_scheduler.step()

        # Validation
        cur_step = (cur_epoch + 1) * len(train_loader)
        top1 = valid_epoch(valid_loader, model, loss_fun, cur_epoch, cur_step, valid_meter)
        
        # Saving
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        save_checkpoint(model, cfg.OUT_DIR, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train_epoch(train_loader, model, optimizer, criterion, cur_epoch, train_meter):

    # TODO: DALI backend support
    # if config.data_loader_type == 'DALI':
    #     len_train_loader = get_train_loader_len(config.dataset.lower(), config.batch_size, is_train=True)
    # else:
    #     len_train_loader = len(train_loader)
    model.train()
    train_meter.iter_tic()
    cur_step = cur_epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('train/lr', cur_lr, cur_step)

    #TODO: DALI backend support
    # if config.data_loader_type == 'DALI':
    #     for cur_iter, data in enumerate(train_loader):
    #         X = data[0]["data"].cuda(non_blocking=True)
    #         y = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    #         if config.cutout_length > 0:
    #             X = cutout_batch(X, config.cutout_length)
    #         train_iter(X, y)
    #         cur_step += 1
    #     train_loader.reset()
    for cur_iter, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if cfg.TRAIN.AUX_WEIGHT > 0.:
            loss += cfg.TRAIN.AUX_WEIGHT * criterion(aux_logits, y)
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()

        top1_err, top5_err = meters.topk_errors(logits, y, [1, 5])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = X.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(top1_err, top5_err, loss, cur_lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        # write to tensorboard
        writer.add_scalar('train/loss', loss, cur_step)
        writer.add_scalar('train/top1_error', top1_err, cur_step)
        writer.add_scalar('train/top5_error', top5_err, cur_step)
        cur_step += 1
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def valid_epoch(valid_loader, model, criterion, cur_epoch, cur_step, valid_meter):
    model.eval()
    valid_meter.iter_tic()
    for cur_iter, (X, y) in enumerate(valid_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = model(X)
        loss = criterion(logits, y)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(logits, y, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        # NOTE: this line is disabled before.
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        valid_meter.iter_toc()
        # Update and log stats
        valid_meter.update_stats(
            top1_err, top5_err, X.size(0) * cfg.NUM_GPUS)
        valid_meter.log_iter_stats(cur_epoch, cur_iter)
        valid_meter.iter_tic()
    top1_err = valid_meter.mb_top1_err.get_win_median()
    valid_meter.log_epoch_stats(cur_epoch)
    valid_meter.reset()
    return top1_err


def get_data(dataset, data_path, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10' or dataset == 'cifar10_24' or dataset == 'cifar10_16':
        dset_cls = dset.CIFAR10
        n_classes = 10
    else:
        raise NotImplementedError

    trn_transform, val_transform = _data_transforms_cifar10(cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    if hasattr(trn_data, 'data'):
        shape = trn_data.data.shape
    else:
        shape = trn_data.train_data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]
    if dataset == 'cifar10_16':
        input_size = 16
    if dataset == 'cifar10_24':
        input_size = 24

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation: # append validation data
        ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret


def _data_transforms_cifar10(cutout_length):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--genotype', required=False, help='Cell genotype')
    # args = parser.parse_args()
    # arg_genotype = args.genotype
    # _genotype = gt.from_str(arg_genotype)
    main()
