import numpy as np
import math as m
import matplotlib.pyplot as plt
import cmac


class AproximateSin:

    def __init__(self):
        self.generate_sets()
        self.generate_cmac()
        self.train()
        self.predict()
        self.plot()
    
    def generate_sets(self):
        domain = np.linspace(-m.pi, m.pi, 1000)
        np.random.shuffle(domain)
        # 50% for training
        training_set = domain[0:500]
        # 50% for test
        test_set = domain[500: 1000]

        training_set.sort()
        test_set.sort()
        
        training_set = training_set.reshape((500,1))
        test_set = test_set.reshape((500,1))

        self._training_set = np.concatenate((training_set,  np.sin(training_set)), axis=1)
        self._test_set = np.concatenate((test_set, np.sin(test_set)), axis=1)

    def generate_cmac(self):
        conf = cmac.SignalConfiguration(-m.pi, m.pi, 100, 'Sin function between -pi and pi')
        ann = cmac.CMAC([conf], 2)
        self._cmac = ann

    def train(self):
        train = cmac.Train(self._cmac, self._training_set[:, 0], self._training_set[:, 1], 0.5, 100)
        train.train()

    def predict(self):
        self._prediction = self._cmac.fire_all(self._test_set)

    def plot(self):
        plt.plot(self._test_set[:,0], self._test_set[:,1], 'r', self._test_set[:,0], self._prediction)
        plt.show()

if __name__ == '__main__':
    AproximateSin()
