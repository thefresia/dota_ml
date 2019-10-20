#%%
from training.train import ModelTraining
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import time
import enlighten
import yaml
import logging
import logging.config

accuracy = lambda x : x['correct']/x['matches']*100
model = None
logger = None

def main():
    load_config()
    trainSkip = 70000
    testSize = 6000
    logger.info("Started")
    global model
    model = ModelTraining(trainSkip, testSize)
    #testOne(0.5,2)
    testRange()
    logger.info("Exiting...")

def load_config():
    with open('logging.yaml', 'r') as f:
        log_cfg = yaml.safe_load(f.read())
        logging.config.dictConfig(log_cfg)
    global logger
    logger = logging.getLogger('dev')


def testOne(activator, multiplier, clusters = 2):
    model.prepare_data(model.trainSkip)
    model.set_params(activator,multiplier)
    return model.clusterize_train(clusters)

def testRange(activatorStart = 0.4, activatorEnd = 0.6, activatorStep = 0.005, 
        multiplierStart = 1.0, multiplierEnd = 2.0, multiplierStep = 0.01):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    activator = np.arange(activatorStart, activatorEnd, activatorStep, dtype=np.dtype(float))
    multiplier = np.arange(multiplierStart, multiplierEnd, multiplierStep, dtype=np.dtype(float))
    X, Y = np.meshgrid(activator, multiplier)
    Z = np.copy(X)
    pbar = enlighten.Counter(total=np.size(Z,1)*np.size(Z,0), desc='Progress', unit='tests')
    for i in range(np.size(Z,1)):
        for j in range(np.size(Z,0)):
            model.set_params(X[0,i].item(), Y[j,i].item())
            trained = model.clusterize_train(2)
            Z[j,i] = accuracy(trained)
            pbar.update()

    surf = ax.plot_surface(X, Y, Z, cmap= cm.get_cmap("coolwarm"), rstride=1, cstride=1,
                        linewidth=0, antialiased=True)

    ax.set_zlim(46, 61)
    ax.set_xlabel('Активатор') 
    ax.set_xticks(activator)
    ax.set_ylabel('Множитель') 
    ax.set_yticks(multiplier)
    ax.set_zlabel('Точность предсказания') 
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.3, aspect=5)

    plt.show()

if __name__ == '__main__':
    main()
