import purecma
import numpy as np
import random
from utils import generate_from_grid_2D, generate_random_points
from surrogate import DecisionTree_Surrogate

import sys

try:
    sys.path.append('./')
    from cec2017.functions import *
except Exception:
    sys.path.append('../')
    from cec2017.functions import *


N=2 # dimensions
dot = lambda X: np.dot(X,X) # x^2+y^2, but for any dimension


def target(x, function=dot):
    # purecma sometimes passes list of floats, sometimes list of lists... 
    if isinstance(x[0],list):
        return list(function(xi) for xi in x)
    else:
        return function(x)


def get_surrogate_grid(seed, size):
    np.random.seed(seed)
    model = DecisionTree_Surrogate(target)
    random.seed(seed)
    X,Y = generate_from_grid_2D(target, size)
    model.train_surrogate(X,Y)
    return model

def get_surrogate_random(seed,size, dimension=2):
    np.random.seed(seed)
    model = DecisionTree_Surrogate(target)
    random.seed(seed)
    X,Y = generate_random_points(target, dimension, size)
    model.train_surrogate(X,Y)
    return model

def get_untrained_surrogate():
    return DecisionTree_Surrogate(target,0)

def experiment1():
    # two cases 1) pure CMA, 2) model that gets used interchangably with real function
    x0 = [-15]*N  # initial solution
    sigma0 = 0.5  # initial standard deviation to sample new solutions

    iterations_with_model = []
    iterations_without_model = []

    fmin_with_model = []
    fmin_without_model = []

    objective_calls_with_model = []
    objective_calls_without_model = []

    # 50 runs like CEC recommends
    for i in range(50):
        # without surrogate
        random.seed(i)
        xopt, es = purecma.fmin(target, x0, sigma0)
        if i==0:
            es.logger.plot(series_name="czyste CMA-ES") # holds on the plot for next call
        iterations_without_model.append(es.logger.counter)
        fmin_without_model.append(es.best.f)
        objective_calls_without_model.append(es.best.evals)

        fmin = es.best.f

         # with surrogate
        np.random.seed(i)
        f = get_untrained_surrogate()
       
        random.seed(i)
        xopt, es = purecma.fmin(f, x0, sigma0)
        if i==0:
            f.plot_true_log(es.logger._data['iter'], fmin)
            es.logger.plot(f"Wykres zbieżności CMA-ES, bez i z modelem (CEC2017 f9)",
                            series_name="z modelem zastępczym", override_fmin=fmin)

        iterations_with_model.append(es.logger.counter)
        fmin_with_model.append(f.get_true_min())
        objective_calls_with_model.append(f.objective_calls)

    print("avg with model:")
    print(f"    fmin: {np.mean(fmin_with_model)}")
    print(f"    iterations: {np.mean(iterations_with_model)}")
    print(f"    objective calls: {np.mean(objective_calls_with_model)}")

    print("avg without model:")
    print(f"    fmin: {np.mean(fmin_without_model)}")
    print(f"    iterations: {np.mean(iterations_without_model)}")
    print(f"    objective calls: {np.mean(objective_calls_without_model)}")



def experiment2():
    # comparison of three cases 1) pure CMA 2) model initialized with a grid/random 3) same model without init
    x0 = [-15]*2  # initial solution
    sigma0 = 0.5  # initial standard deviation to sample new solutions
    iterations_with_model = []
    iterations_without_model = []

    fmin_with_model = []
    fmin_without_model = []

    objective_calls_with_model = []
    objective_calls_without_model = []

    # 50 runs like CEC recommends
    for i in range(50):
        
        random.seed(i)
        xopt, es = purecma.fmin(target, x0, sigma0)
        if i==0:
            es.logger.plot(series_name="czyste CMA-ES") # holds on the plot for next call
        iterations_without_model.append(es.logger.counter)
        fmin_without_model.append(es.best.f)
        objective_calls_without_model.append(es.best.evals)

        fmin = es.best.f

        #print("normal")
        # get new surrogate
        np.random.seed(i)
        f = get_untrained_surrogate()
        random.seed(i)
        xopt, es = purecma.fmin(f, x0, sigma0)
        if i==0:
            f.plot_true_log(es.logger._data['iter'], fmin)
            es.logger.plot(series_name="z modelem zastępczym", override_fmin=fmin)

        print("grid")
        f = get_surrogate_grid(i, 400)
        #f = get_surrogate_random(i,200*200)
        random.seed(i)
        xopt, es = purecma.fmin(f, x0, sigma0)
        if i==0:
            f.plot_true_log(es.logger._data['iter'], fmin)
            es.logger.plot(f"Wykres zbieżności CMA-ES, bez i z modelem (np.dot)",
                            series_name="stały model i siatka inicjalizacyjna", override_fmin=fmin)

        iterations_with_model.append(es.logger.counter)
        fmin_with_model.append(f.get_true_min())
        objective_calls_with_model.append(f.objective_calls)
    print("avg with model:")
    print(f"    fmin: {np.mean(fmin_with_model)}")
    print(f"    iterations: {np.mean(iterations_with_model)}")
    print(f"    objective calls: {np.mean(objective_calls_with_model)}")

    print("avg without model:")
    print(f"    fmin: {np.mean(fmin_without_model)}")
    print(f"    iterations: {np.mean(iterations_without_model)}")
    print(f"    objective calls: {np.mean(objective_calls_without_model)}")

if __name__ == "__main__":
    experiment2()