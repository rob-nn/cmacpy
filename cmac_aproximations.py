import timeit
import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cmacpy


class AproximateSin:

    def __init__(self):
        self.generate_sets()
        self.generate_cmacpy()
        self.train()
        self.predict()
    
    
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

    def generate_cmacpy(self):
        conf = cmacpy.SignalConfiguration(-m.pi, m.pi, 100, 'Sin function between -pi and pi')
        ann = cmacpy.CMAC([conf], 2)
        self._cmacpy = ann

    def train(self):
        train = cmacpy.Train(self._cmacpy, self._training_set[:, 0], self._training_set[:, 1], 0.5, 100)
        train.train()

    def predict(self):
        self._prediction = self._cmacpy.fire_all(self._test_set)

    def plot(self):
        plt.plot(self._test_set[:,0], self._test_set[:,1], 'r', self._test_set[:,0], self._prediction)
        plt.show()

class AproximateSombrero:
    def __init__(self):
        self.generate_sets()
        self.generate_cmacpy()
        self.train()
        self.predict()
        self.plot()

    def generate_sets(self):
        self._num_elements = 300 
        x = np.linspace(-10, 10, self._num_elements *2)
        y = np.linspace(-10, 10, self._num_elements *2)
        np.random.shuffle(x)
        np.random.shuffle(y)
        x_train= x[0:self._num_elements]
        x_test= x[self._num_elements: 2*self._num_elements]
        x_train.sort()
        x_test.sort()
        y_train= x[0:self._num_elements]
        y_test= x[self._num_elements: 2*self._num_elements]
        y_train.sort()
        y_test.sort()
        self._training_set = np.concatenate((x_train.reshape((self._num_elements, 1)), y_train.reshape((self._num_elements, 1))), axis=1)
        self._test_set = np.concatenate((x_test.reshape((self._num_elements, 1)), y_test.reshape((self._num_elements, 1))), axis=1)
        
    def generate_cmacpy(self):
        x_axis = cmacpy.SignalConfiguration(-10, 10, 100, 'X axis')
        y_axis = cmacpy.SignalConfiguration(-10, 10, 100, 'Y axis')
        self._cmacpy = cmacpy.CMAC([x_axis, y_axis], 5)

    def train(self):
        input_values, output_values = self._sombrero_mesh_shaped(self._training_set)
        train = cmacpy.Train(self._cmacpy, input_values, output_values, 0.5, 30)
        train.train()
    
    def predict(self):
        self._test_input_values, self._test_output_values = self._sombrero_mesh_shaped(self._test_set)
        self.prediction = self._cmacpy.fire_all(self._test_input_values)

    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(self._test_input_values[:,0].reshape((self._num_elements, self._num_elements)), self._test_input_values[:,1].reshape((self._num_elements, self._num_elements)) , self._test_output_values.reshape((self._num_elements, self._num_elements)), cmap=cm.coolwarm)
        plt.show()        

        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        ax2.plot_surface(self._test_input_values[:,0].reshape((self._num_elements, self._num_elements)), self._test_input_values[:,1].reshape((self._num_elements, self._num_elements)) , np.array(self.prediction).reshape((self._num_elements, self._num_elements)), cmap=cm.coolwarm)

        plt.show()

    def _sombrero_mesh_shaped(self, data_set):
        x = data_set[:, 0]
        y = data_set[:, 1]
        x,y = np.meshgrid(x, y)
        r = np.sqrt(x ** 2 + y ** 2)
        z = np.sin(r) / r
        x = x.reshape(self._num_elements ** 2, 1)
        y = y.reshape(self._num_elements ** 2, 1)
        z = z.reshape(self._num_elements ** 2, 1)
        input_values = np.concatenate(((x, y)), axis=1)
        return input_values, z
if __name__ == '__main__':
    #AproximateSin()
    AproximateSombrero()
