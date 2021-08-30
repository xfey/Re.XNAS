"""
feature extractor
    calculate features for given architecture
input:
    genotype
    weights
output:
    features matrix
    similar matrix
"""

"""train indepandent (can be augmented) model for DARTS"""
"""only support cifar10 now"""


import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import cka
import xnas.core.checkpoint as checkpoint
import xnas.core.config as config
import xnas.core.distributed as dist
import xnas.core.logging as logging
import xnas.core.meters as meters
from xnas.core.config import cfg
from xnas.core.builders import build_loss_fun
from xnas.core.trainer import setup_env
from xnas.search_space.cellbased_DARTS_cnn import AugmentCNN
from xnas.datasets.loader import construct_loader
device = torch.device("cuda")

writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))

logger = logging.get_logger(__name__)

# Load config and check
config.load_cfg_fom_args()
config.assert_and_infer_cfg()
cfg.freeze()


def random_data_gen(data_loader):
    target_i = random.randint(0, 620)
    for i, (trn_X, trn_y) in enumerate(data_loader):
        if i == target_i:
            return (trn_X, trn_y)


def main():
    setup_env()

    # 32 3 10 === 32 16 10
    # print(input_size, input_channels, n_classes, '===', cfg.SEARCH.IM_SIZE, cfg.SPACE.CHANNEL, cfg.SEARCH.NUM_CLASSES)

    loss_fun = build_loss_fun().cuda()
    use_aux = cfg.TRAIN.AUX_WEIGHT > 0.

    # SEARCH.INIT_CHANNEL as 3 for rgb and TRAIN.CHANNELS as 32 by manual.
    # IM_SIZE, CHANNEL and NUM_CLASSES should be same with search period.
    model = AugmentCNN(cfg.SEARCH.IM_SIZE, cfg.SEARCH.INPUT_CHANNEL, cfg.TRAIN.CHANNELS,
                       cfg.SEARCH.NUM_CLASSES, cfg.TRAIN.LAYERS, use_aux, cfg.TRAIN.GENOTYPE)

    # TODO: Parallel
    # model = nn.DataParallel(model, device_ids=cfg.NUM_GPUS).to(device)
    model.cuda()

    # Get data loader
    [_train_data, data_loader] = construct_loader(
        cfg.TRAIN.DATASET, cfg.TRAIN.SPLIT, cfg.TRAIN.BATCH_SIZE)

    checkpoint.load_checkpoint("tools/itpruner/test.pyth", model)
    logger.info("load checkpoint done.")

    (data_X, data_y) = random_data_gen(data_loader)
    data_X, data_y = data_X.cuda(), data_y.cuda()

    # print(model(data_X))

    with torch.no_grad():
        feature = model.feature_extractor(data_X)

    for i in feature:
        print(i.size())
    
    # data_X.size()[0] = cfg.TRAIN.BATCH_SIZE

    for i in range(len(feature)):
        feature[i] = feature[i].view(data_X.size()[0], -1)
        feature[i] = feature[i].data.cpu().numpy()

    similar_matrix = np.zeros((len(feature), len(feature)))

    for i in range(len(feature)):
        for j in range(len(feature)):
            with torch.no_grad():
                similar_matrix[i][j] = cka.cka(cka.gram_linear(feature[i]), cka.gram_linear(feature[j]))

    print(similar_matrix)


if __name__ == "__main__":
    main()
