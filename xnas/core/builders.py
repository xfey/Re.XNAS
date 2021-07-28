# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

import torch

from xnas.core.config import cfg

from xnas.search_space.cellbased_DARTS_cnn import _DartsCNN
from xnas.search_space.cellbased_PDARTS_cnn import _PdartsCNN
from xnas.search_space.cellbased_PCDARTS_cnn import _PcdartsCNN
from xnas.search_space.cellbased_NASBench201_cnn import _NASBench201
from xnas.search_space.cellbased_1shot1_cnn import _NASbench1shot1_1, _NASbench1shot1_2, _NASbench1shot1_3

from xnas.search_space.mb_v3_cnn import build_super_net

# Supported models
_spaces = {
    "darts": _DartsCNN,
    "pdarts": _PdartsCNN,
    "pcdarts": _PcdartsCNN,
    "ofa": build_super_net,
    "google": build_super_net,
    "proxyless": build_super_net,
    "nasbench201": _NASBench201,
    "nasbench1shot1_1": _NASbench1shot1_1,
    "nasbench1shot1_2": _NASbench1shot1_2,
    "nasbench1shot1_3": _NASbench1shot1_3
}

# Supported loss functions
_loss_funs = {
    "cross_entropy": torch.nn.CrossEntropyLoss
}


def get_space():
    """Get the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.SPACE.NAME in _spaces.keys(), err_str.format(cfg.SPACE.NAME)
    return _spaces[cfg.SPACE.NAME]


def get_loss_fun():
    """Get the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_space():
    """Build the model."""
    return get_space()()


def build_loss_fun():
    """Build the loss function."""
    return get_loss_fun()()


def register_space(name, ctor):
    """Register a model dynamically."""
    _spaces[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor


def sng_builder(category):
    # TODO: finish SNG search
    raise NotImplementedError


def lr_scheduler_builder(w_optim):
    if cfg.OPTIM.LR_POLICY == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, cfg.OPTIM.MAX_EPOCH, eta_min=cfg.OPTIM.MIN_LR)
    elif cfg.OPTIM.LR_POLICY == "step":
        return torch.optim.lr_scheduler.MultiStepLR(w_optim, cfg.OPTIM.STEPS, gamma=cfg.OPTIM.LR_MULT)
    else:
        raise NotImplementedError