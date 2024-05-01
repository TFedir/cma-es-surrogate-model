import cma_pop
import numpy as np
from surrogate import DecisionTree_Surrogate
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys

try:
    sys.path.append('./')
    from cec2017.functions import *
except Exception:
    sys.path.append('../')
    from cec2017.functions import *

def calc_acc(func, x_test, y_test):
    res = []
    for x in x_test:
        res.append(func.surrogate_pred([x]))
    return r2_score(y_test, res)


def generate_random_points(target, args_amount, amount):
    args = [
        [random.uniform(-100, 100) for _ in range(args_amount)] for _ in range(amount)
    ]
    result = []
    for arg in args:
        result.append(target(arg))
    return (args, result)


def generate_from_grid_2D(target, amount):
    args = []
    for x in np.linspace(-100, 100, amount):
        for y in np.linspace(-100, 100, amount):
            args.append([x,y])
    result = []
    for arg in args:
        result.append(target(arg))
    return (args, result)
