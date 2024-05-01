from sklearn import tree
import numpy as np

import matplotlib.pyplot as plt

class DecisionTree_Surrogate:
    def __init__(self, target, model_start=1000) -> None:
        self.target = target
        self.surrogate = tree.DecisionTreeRegressor()
        self.use_surrogate = True
        self.log = []

        self.train_data_X = []
        self.train_data_Y = []

        self.objective_calls=0

        # despite using model, the real values will be recorded as well
        # to create plot of true result 
        self.true_log = [0] 

        self.calls = 0

        self.model_start = model_start

    def surrogate_pred(self, x):
        return self.surrogate.predict(x)

    def train_surrogate(self, X, Y):
        # train model, and remember initial points for retraining
        # X - list of lists (or 2d np.array)
        # Y - list of correct caclues

        if not self.train_data_X:
            # remember data for future trainings
            self.train_data_X = X
            self.train_data_Y = Y
        self.surrogate = self.surrogate.fit(X, Y)

    def predictor(self,X):
        # works as a middleman between real objective function and solver,
        # experiment logic is held here (training every nth step etc)

        self.calls+=len(X)
        if isinstance(X[0],float):
            # pack it as a single sample (sometimes library gives this format)
            X = [X]
        if self.use_surrogate:
                #self.use_surrogate = not self.use_surrogate
                pred = self.surrogate_pred(X)
                # --- for statistics, don't count evaluation
                true_value = self.target(X)
                self.true_log.append(np.min(true_value))
                # ---
                return pred
        else:
            self.objective_calls+=len(X)
            true_value = self.target(X)
            self.true_log.append(np.min(true_value))
            self.train_data_X.extend(X)
            self.train_data_Y.extend(true_value)

            if len(self.train_data_X)>=self.model_start and not len(self.train_data_X)%10:
                print(f"training on {len(self.train_data_X)}")
                # every 10 samples, use it
                self.use_surrogate = True
                # experiment 1
                #if len(self.train_data_X)>1000:
                #    self.train_data_X = self.train_data_X[-1000:]
                #    self.train_data_Y = self.train_data_Y[-1000:]
                self.surrogate.fit(self.train_data_X, self.train_data_Y)
            return true_value
    
    def plot_true_log(self, iters, fmin):
        self.true_log.pop() # one value offsets the plot, because we collect log by hand
        self.true_log[0]=self.true_log[1]
        plt.plot(iters,self.true_log-fmin, label="faktyczna ewaluacja f.celu", linewidth=1)

    def get_true_min(self):
        return np.min(self.true_log[1:])

    def __call__(self,X):
        return self.predictor(X)
