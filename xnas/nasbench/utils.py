from xnas.nasbench.nasbench1shot1 import Eval_nasbench1shot1
from xnas.nasbench.nasbench201 import Eval_nasbench201


def EvaluateNasbench(theta, search_space, logger, NASbenchName):
    """evaluate nasbench by import different functions."""
    
    if NASbenchName in ["nasbench1shot1_1", "nasbench1shot1_2", "nasbench1shot1_3"]:
        Eval_nasbench1shot1(theta, search_space, logger)
    elif NASbenchName == "nasbench201":
        Eval_nasbench201(theta, search_space, logger)
