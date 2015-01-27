import unittest
import cmac
from numpy import *
import numpy as np
import random

class TestCmac(unittest.TestCase):

    def setUp(self):
        self._sense_conf = cmac.SignalConfiguration(0., 12., 13)
        self._sense_conf_2 = cmac.SignalConfiguration(0.1, 1., 10)
        self._sense_conf_3 = cmac.SignalConfiguration(-1, 1., 2)
        confs = []
        confs.append(self._sense_conf)
        confs.append(self._sense_conf_2)
	confs.append(self._sense_conf_3)
        self._cmac = cmac.CMAC(confs, 4)

    def tearDown(self):
        self._sense_conf = None

    def test_get_layer1_vector(self):
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(0) == np.array([0, 1, 2, 3])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(10) == np.array([12, 13, 10, 11])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(12) == np.array([12, 13, 14, 15])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(0) == np.array([0, 1, 2, 3])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(6) == np.array([8, 9, 6, 7])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(10) == np.array([12, 13, 10, 11])))
        self.assertTrue(np.all(self._sense_conf_3.get_layer1_vector(0) == np.array([0, 1, 2, 3])))
        self.assertTrue(np.all(self._sense_conf_3.get_layer1_vector(1) == np.array([4, 1, 2, 3])))
        self.assertTrue(np.all(self._sense_conf_3.get_layer1_vector(2) == np.array([4, 5, 2, 3])))

    def test_get_weight_table(self):
        self.assertTrue(self._cmac.get_weight(41212) != 0)
        self.assertTrue(self._cmac.get_weight(0) != 0)
        self.assertTrue(self._cmac.get_weight(10109) != 0)
        self.assertTrue(self._cmac.get_weight(30307) != 0)

        self.assertTrue(self._cmac.get_weight(-1) != 0)

    def test_get_weight_table(self):
        self._cmac.set_weight(41212, 55)
        self.assertTrue(self._cmac.get_weight(41212) == 55)

    def test_max_value(self):
        self.assertTrue(self._sense_conf.max_value == 15)
        self.assertTrue(self._sense_conf_2.max_value == 12)
        self.assertTrue(self._sense_conf_3.max_value == 4)

    def test_discret_values_first_value(self):
        self.assertTrue(self._sense_conf.discret_values[0] == self._sense_conf.s_min)

    def test_discret_values_last_value(self):
        self.assertTrue(self._sense_conf.discret_values[-1] == self._sense_conf.s_max)
   
    def test_discret_values_size(self):
    	self.assertTrue(len(self._sense_conf.discret_values) == 13)

    def test_discret_values_all(self):
	self.assertTrue( \
		(around(self._sense_conf.discret_values,0) == \
		around(array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]),0)).all())
	self.assertTrue( \
		(around(self._sense_conf_2.discret_values.tolist(), 1) == \
		around([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], 1)).all())
    def test_get_discretized_value_index(self):
	self.assertTrue(self._cmac.get_discretized_value_index(0.25, self._sense_conf_2) == 1)
	self.assertTrue(self._cmac.get_discretized_value_index(1., self._sense_conf_2) == 9)
	self.assertTrue(self._cmac.get_discretized_value_index(5, self._sense_conf_2) == 9)
	self.assertTrue(self._cmac.get_discretized_value_index(0.1, self._sense_conf_2) == 0)
	self.assertTrue(self._cmac.get_discretized_value_index(0, self._sense_conf_2) == 0)
	self.assertTrue(self._cmac.get_discretized_value_index(0.95, self._sense_conf_2) == 8)
	self.assertTrue(self._cmac.get_discretized_value_index(0.4, self._sense_conf_2) == 3)


    def test_mapping_lines_sens_conf2(self):
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(0) == np.array([0, 1, 2, 3])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(1) == np.array([4, 1, 2, 3])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(2) == np.array([4, 5, 2, 3])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(3) == np.array([4, 5, 6, 3])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(4) == np.array([4, 5, 6, 7])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(5) == np.array([8, 5, 6, 7])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(6) == np.array([8, 9, 6, 7])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(7) == np.array([8, 9, 10, 7])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(8) == np.array([8, 9, 10, 11])))
        self.assertTrue(np.all(self._sense_conf_2.get_layer1_vector(9) == np.array([12, 9, 10 , 11])))
 
    def test_mapping_all_lines_sens_conf(self):
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(0) == np.array([0, 1, 2, 3])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(1) == np.array([4, 1, 2, 3])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(2) == np.array([4, 5, 2, 3])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(3) == np.array([4, 5, 6, 3])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(4) == np.array([4, 5, 6, 7])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(5) == np.array([8, 5, 6, 7])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(6) == np.array([8, 9, 6, 7])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(7) == np.array([8, 9, 10, 7])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(8) == np.array([8, 9, 10, 11])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(9) == np.array([12, 9, 10 , 11])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(10) == np.array([12, 13, 10 , 11])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(11) == np.array([12, 13, 14 , 11])))
        self.assertTrue(np.all(self._sense_conf.get_layer1_vector(12) == np.array([12, 13, 14 , 15])))

    def test_weight_table(self):
	self.assertTrue(np.all(self._cmac.calculate_activation_adresses([12., 1., 1.])== np.array([41212, 10913, 21014, 31115])))
	self.assertTrue(np.all(self._cmac.calculate_activation_adresses([0,0,0])== np.array([0, 10101, 20202, 30303])))
	self.assertTrue(np.all(self._cmac.calculate_activation_adresses([5.25, 0.37, -0.2])== np.array([408, 10505, 20206, 30307])))
	self.assertTrue(self._cmac.get_weight(41212) != 0)
	self.assertTrue(self._cmac.get_weight(10913) != 0)
	self.assertTrue(self._cmac.get_weight(21014) != 0)
	self.assertTrue(self._cmac.get_weight(31115) != 0)
	self.assertTrue(self._cmac.get_weight(10101) != 0)
	self.assertTrue(self._cmac.get_weight(20202) != 0)
	self.assertTrue(self._cmac.get_weight(30303) != 0)
	self.assertTrue(self._cmac.get_weight(408) != 0)
	self.assertTrue(self._cmac.get_weight(10505) != 0)
	self.assertTrue(self._cmac.get_weight(20206) != 0)
	self.assertTrue(self._cmac.get_weight(30307) != 0)
	self.assertTrue(self._cmac.fire([4, 0.3, -0.5]) != 0)
 

class TestTrain(unittest.TestCase):
    def setUp(self):
        confs = []
        confs.append(cmac.SignalConfiguration(-10., 10., 100))
        confs.append(cmac.SignalConfiguration(-10., 10., 100))
        _cmac = cmac.CMAC(confs, 4)
        data_in = None
        data_out = array([])
        for i in range(100):
            n1 = random.uniform(-100, 100)
            n2 = random.uniform(-100, 100)
            temp = array([[n1, n2]])
            if data_in == None: data_in = temp
            else: data_in = concatenate((data_in, temp))
            data_out = concatenate((data_out, array([random.uniform(-100, 100)])))
        data_out = reshape(data_out, (len(data_out), 1))
        self._train = cmac.Train(_cmac, data_in, data_out, 0.5, 10)
        
        
    
    def test_train(self): 
        self._train.train()
        
   
def main():
    unittest.main()

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCmac)
    unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
    main()
