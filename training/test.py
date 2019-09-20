import numpy as np

class Test:
    modelSize = 0
    clustersSize = 0
    trainSize = 0
    
    def __init__(self):
        pass

    def train_test(self, size, clusters, trainSize):
        self.modelSize = size
        self.trainSize = trainSize
        self.clustersSize = clusters
        return self.clustersSize*10

m = Test()
X = np.arange(4, 9, 1)
Y = np.arange(50, 70, 10)
X, Y = np.meshgrid(X, Y)
Z = m.train_test(Y,X,60)
print('byeh')