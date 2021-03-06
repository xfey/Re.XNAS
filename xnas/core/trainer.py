# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import gc
import os
import sys

import numpy as np
import torch
import random

import xnas.core.config as config
import xnas.core.distributed as dist
import xnas.core.logging as logging
import xnas.core.meters as meters
from xnas.core.config import cfg


logger = logging.get_logger(__name__)


def setup_env():
    """Set up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure the output dir exists and save config
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    if cfg.DETERMINSTIC:
        # Fix RNG seeds
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        # Configure the CUDNN backend
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


# def setup_model():
#     """Sets up a model for training or testing and log the results."""
#     # Build the model
#     model = builders.build_space()
#     logger.info("Model:\n{}".format(model))
#     # Log model complexity
#     logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
#     # Transfer the model to the current GPU device
#     err_str = "Cannot use more GPU devices than available"
#     assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
#     cur_device = torch.cuda.current_device()
#     model = model.cuda(device=cur_device)
#     # Use multi-process data parallel model in the multi-gpu setting
#     if cfg.NUM_GPUS > 1:
#         # Make model replica operate on the current device
#         model = torch.nn.parallel.DistributedDataParallel(
#             module=model, device_ids=[cur_device], output_device=cur_device
#         )
#         # Set complexity function to be module's complexity function
#         model.complexity = model.module.complexity
#     return model


# def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
#     """Performs one epoch of training."""
#     # Shuffle the data
#     loader.shuffle(train_loader, cur_epoch)
#     # Update the learning rate
#     lr = optim.get_epoch_lr(cur_epoch)
#     optim.set_lr(optimizer, lr)
#     # Enable training mode
#     model.train()
#     train_meter.iter_tic()
#     # scale the grad in amp, amp only support the newest version
#     scaler = torch.cuda.amp.GradScaler() if cfg.SEARCH.AMP & hasattr(
#         torch.cuda.amp, 'autocast') else None
#     for cur_iter, (inputs, labels) in enumerate(train_loader):
#         # Transfer the data to the current GPU device
#         inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
#         # using AMP
#         if scaler is not None:
#             with torch.cuda.amp.autocast():
#                 # Perform the forward pass in AMP
#                 preds = model(inputs)
#                 # Compute the loss in AMP
#                 loss = loss_fun(preds, labels)
#                 # Perform the backward pass in AMP
#                 optimizer.zero_grad()
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 # Updates the scale for next iteration.
#                 scaler.update()
#         else:
#             preds = model(inputs)
#             # Compute the loss
#             loss = loss_fun(preds, labels)
#             # Perform the backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             # Update the parameters
#             optimizer.step()
#         # Compute the errors
#         top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
#         # Combine the stats across the GPUs (no reduction when 1 GPU)
#         loss, top1_err, top5_err = dist.scaled_all_reduce(
#             [loss, top1_err, top5_err])
#         # Copy the stats from GPU to CPU (sync point)
#         loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
#         train_meter.iter_toc()
#         # Update and log stats
#         mb_size = inputs.size(0) * cfg.NUM_GPUS
#         train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
#         train_meter.log_iter_stats(cur_epoch, cur_iter)
#         train_meter.iter_tic()
#     # Log epoch stats
#     train_meter.log_epoch_stats(cur_epoch)
#     train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch, tensorboard_writer=None):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # using AMP
        if cfg.SEARCH.AMP & hasattr(torch.cuda.amp, 'autocast'):
            with torch.cuda.amp.autocast():
                # Compute the predictions
                preds = model(inputs)
        else:
            # Compute the predictions
            preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        # NOTE: this line is disabled before.
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    top1_err = test_meter.mb_top1_err.get_win_median()
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar(
            'val/top1_error', test_meter.mb_top1_err.get_win_median(), cur_epoch)
        tensorboard_writer.add_scalar(
            'val/top5_error', test_meter.mb_top5_err.get_win_median(), cur_epoch)
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()
    return top1_err


# def train_model():
#     """Trains the model."""
#     # Setup training/testing environment
#     setup_env()
#     # Construct the model, loss_fun, and optimizer
#     model = setup_model()
#     loss_fun = builders.build_loss_fun().cuda()
#     optimizer = optim.construct_optimizer(model)
#     # Load checkpoint or initial weights
#     start_epoch = 0
#     if cfg.SEARCH.AUTO_RESUME and checkpoint.has_checkpoint():
#         last_checkpoint = checkpoint.get_last_checkpoint()
#         checkpoint_epoch = checkpoint.load_checkpoint(
#             last_checkpoint, model, optimizer)
#         logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
#         start_epoch = checkpoint_epoch + 1
#     elif cfg.SEARCH.WEIGHTS:
#         checkpoint.load_checkpoint(cfg.SEARCH.WEIGHTS, model)
#         logger.info("Loaded initial weights from: {}".format(
#             cfg.SEARCH.WEIGHTS))
#     # Create data loaders and meters
#     [train_loader, test_loader] = loader.construct_loader(
#         cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)
#     train_meter = meters.TrainMeter(len(train_loader))
#     test_meter = meters.TestMeter(len(test_loader))
#     # Compute model and loader timings
#     if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
#         benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
#     # Perform the training loop
#     logger.info("Start epoch: {}".format(start_epoch + 1))
#     for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
#         # Train for one epoch
#         train_epoch(train_loader, model, loss_fun,
#                     optimizer, train_meter, cur_epoch)
#         # # Compute precise BN stats
#         # if cfg.BN.USE_PRECISE_STATS:
#         #     net.compute_precise_bn_stats(model, train_loader)
#         # Save a checkpoint
#         if (cur_epoch + 1) % cfg.SEARCH.CHECKPOINT_PERIOD == 0:
#             checkpoint_file = checkpoint.save_checkpoint(
#                 model, optimizer, cur_epoch)
#             logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
#         # Evaluate the model
#         next_epoch = cur_epoch + 1
#         if next_epoch % cfg.SEARCH.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
#             logger.info("Start testing")
#             test_epoch(test_loader, model, test_meter, cur_epoch)
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#             torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
#         gc.collect()


# def test_model():
#     """Evaluates a trained model."""
#     # Setup training/testing environment
#     setup_env()
#     # Construct the model
#     model = setup_model()
#     # Load model weights
#     checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
#     logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
#     # Create data loaders and meters
#     # test_loader = loader.construct_test_loader()
#     [train_loader, test_loader] = loader.construct_loader(
#         cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)
#     test_meter = meters.TestMeter(len(test_loader))
#     # Evaluate the model
#     test_epoch(test_loader, model, test_meter, 0)


# def time_model():
#     """Times model and data loader."""
#     # Setup training/testing environment
#     setup_env()
#     # Construct the model and loss_fun
#     model = setup_model()
#     loss_fun = builders.build_loss_fun().cuda()
#     # Create data loaders
#     [train_loader, test_loader] = loader.construct_loader(
#         cfg.SEARCH.DATASET, cfg.SEARCH.SPLIT, cfg.SEARCH.BATCH_SIZE)
#     # Compute model and loader timings
#     benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
