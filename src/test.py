from utils import generate_from_grid_2D, generate_random_points, calc_acc
from surrogate import DecisionTree_Surrogate
import random, numpy as np
from sklearn.model_selection import train_test_split

def test_generation_technique():
    target = lambda X: np.dot(X,X) # x^2+y^2, but for any dimension
    data1 = generate_from_grid_2D(target, 200)
    data2 = generate_random_points(target, 2, 200)
    func = DecisionTree_Surrogate(target)
    x_train, x_test, y_train, y_test = train_test_split(data1[0], data1[1], test_size=0.2, random_state=20)
    func.train_surrogate(x_train, y_train)
    print(f"Grid generation: {calc_acc(func, x_test, y_test)}")
    func_2 = DecisionTree_Surrogate(target)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(data2[0], data2[1], test_size=0.2, random_state=20)
    func_2.train_surrogate(x_train2, y_train2)
    print(f"Random generation: {calc_acc(func_2, x_test2, y_test2)}")

test_generation_technique()