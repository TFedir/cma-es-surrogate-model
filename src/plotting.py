import numpy as np
import matplotlib.pyplot as plt

plt.ion()

def draw_contour(f,d:list[list[float,float]],xrange: tuple[float,float],yrange: tuple[float,float], best_count):
        d = np.array(d)
        x = np.linspace(*xrange, 100)
        y = np.linspace(*yrange, 100)
        X, Y = np.meshgrid(x, y)

        # Evaluate the function at each point in the grid
        Z = []
        for xi in x:
            row = []
            for yi in y:
                  row.append(f([xi,yi]))
            Z.append(row)

        # Create a contour plot
        contour_plot = plt.contour(X, Y, Z, cmap='viridis')
        plt.colorbar(contour_plot, label='Wartosc funkcji')

        d_sorted = get_n_best_by_target(d,f,best_count)
        plt.scatter(d[:,0],d[:,1])
        plt.xlim(*xrange)
        plt.ylim(*yrange)
        plt.scatter(d_sorted[:,0],d_sorted[:,1])
        plt.show()
        plt.pause(0.01)
        plt.clf()

        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('CMA-ES')


def get_n_best_by_target(population, target, n):
    population = np.array(population)
    q = [target(p) for p in population]
    permute = np.argsort(q)[:n] # get n best
    return population[permute]

def split_X(X):
      return np.array(X)[:,0],np.array(X)[:,1]