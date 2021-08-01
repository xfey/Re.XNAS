import random
import string
import time
import sys
import os
import ConfigSpace
import numpy as np

from xnas.core.config import cfg

from xnas.search_space.cellbased_1shot1_ops import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3, OUTPUT_NODE
from nas_201_api import NASBench201API as API201
from nasbench import api


nasbench201_path = 'benchmark/NAS-Bench-201-v1_0-e61699.pth'
nasbench1shot1_path = 'benchmark/nasbench_full.tfrecord'
api_nasben201 = API201(nasbench201_path, verbose=False)
nasbench = api.NASBench(nasbench1shot1_path)


def random_time_string(stringLength=8):
    letters = string.ascii_lowercase
    return str(time.time()).join(random.choice(letters) for i in range(stringLength))


def one_hot_to_index(one_hot_matrix):
    return np.array([np.where(r == 1)[0][0] for r in one_hot_matrix])


def index_to_one_hot(index_vector, C):
    return np.eye(C)[index_vector.reshape(-1)]


def EvaluateNasbench(theta, search_space, logger, NASbenchName):
    # get result log
    stdout_backup = sys.stdout
    result_path = os.path.join(cfg.OUT_DIR, "result.log")
    log_file = open(result_path, "w")
    sys.stdout = log_file
    if NASbenchName == "nasbench201":
        geotype = search_space.genotype(theta)
        index = api_nasben201.query_index_by_arch(geotype)
        api_nasben201.show(index)
    else:
        current_best = np.argmax(theta, axis=1)
        config = ConfigSpace.Configuration(
            search_space.search_space.get_configuration_space(), vector=current_best)
        adjacency_matrix, node_list = search_space.search_space.convert_config_to_nasbench_format(
            config)
        node_list = [INPUT, *node_list, OUTPUT] if search_space.search_space.search_space_number == 3 else [
            INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = nasbench.query(model_spec, epochs=108)
        print("the test accuracy in {}".format(NASbenchName))
        print(nasbench_data['test_accuracy'])

    log_file.close()
    sys.stdout = stdout_backup
