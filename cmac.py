import numpy as np

class CMAC(object):
    def __init__(self, signal_configurations, num_active_cells):
        self._num_active_cells = num_active_cells
        self._signal_configurations = signal_configurations
        self._set_signal_mappings()
        self._table = {}
        np.random.seed()

    def _set_signal_mappings(self):
        for signal_configuration in self._signal_configurations:
            signal_configuration.cmac = self
        self._num_dig = [0]
 
        for conf in self.signal_configuration[0:-1]:
            self._num_dig.append(self._num_dig[-1]+ int(np.floor(np.log10(conf.max_value)+1)))
        self._num_dig = 10 ** np.array(self._num_dig)

    def calculate_address(self, adress_vector):
        return np.dot(self._num_dig, adress_vector)
 
    def calculate_activation_adresses(self, input_signals_vector):
        signals_mapping_values = self._get_signals_mapping_values(input_signals_vector)
        calculated_adresses_vector = []
        for i in range(self.num_active_cells):
            temp = signals_mapping_values[:, i].tolist() #signal concatenation
            address = self.calculate_address(temp)
            calculated_adresses_vector.append(address)
        return np.array(calculated_adresses_vector)
    
    def _get_signals_mapping_values(self, input_signal_vector):
        signals_mapping_values = []
        for i in range(len(self.signal_configuration)):
            index = self.get_discretized_value_index(input_signal_vector[i], self.signal_configuration[i])
            signals_mapping_values.append(self.signal_configuration[i].get_layer1_vector(index))
        return np.array(signals_mapping_values, dtype=np.int64)
    
    def get_discretized_value_index(self, value, config):
        if value <= config.s_min:
            return 0;
        if value >= config.s_max:
            return len(config.discret_values) -1
        for i in range(len(config.discret_values)):
            if config.discret_values[i] > value:
                return i-1 

    def fire(self, input_signal_vector):
        calculated_address_vector = self.calculate_activation_adresses(input_signal_vector)
        p = 0
        for address in calculated_address_vector:
            p = p + self.get_weight(address)    
        return p

    def get_weight(self, address):
        try:
            return self.weight_table[address]
        except KeyError:
            self.weight_table[address] = np.random.uniform(-0.2, 0.2)
            return self.weight_table[address]

    def set_weight(self, address, value):
        self.weight_table[address] = value
            
    @property
    def weight_table(self):
        return self._table            

    @property
    def num_active_cells(self):
        return self._num_active_cells

    @property
    def signal_configuration(self):
        return self._signal_configurations
    
    @property
    def hyperplane(self):
        return self._hyperplane


class Train(object):
    def __init__(self, cmac, data_in, data_out, alpha, num_iterations):
        self._cmac = cmac
        self._data_in = data_in
        self._data_out = data_out
        self._alpha = alpha
        self._num_iterations = num_iterations

    def train(self):
	self.E = []
        for iteration in range(self._num_iterations):
	    err =0
            for i in range(len(self.data_in)):
                input_signal_vector = self.data_in[i, :].tolist()
                out = self.cmac.fire(input_signal_vector)
                for calculated_adress in self.cmac.calculate_activation_adresses(input_signal_vector):
                    self.cmac.set_weight(calculated_adress, self.cmac.get_weight(calculated_adress) + self.alpha * (self.data_out[i] - out) / self.cmac.num_active_cells)
		    err = err+ ((self.data_out[i] - out)**2)/2
	    self.E.append(err)
           
    @property 
    def alpha(self):
        return self._alpha 

    @property
    def data_in(self):
        return self._data_in

    @property
    def data_out(self):
        return self._data_out
    
    @property
    def cmac(self):
        return self._cmac

class SignalConfiguration(object):
    def __init__(self, s_min, s_max, num_discret_values, desc=None):
        self._s_min = s_min
        self._s_max = s_max
        self._num_discret_values = num_discret_values
        self._cmac = None
        self._desc = desc
        self._signal_mapping = None
        self._set_discret_values()

    def _set_discret_values(self):
        self._discret_values = np.linspace(self._s_min, self._s_max, self._num_discret_values)

    def get_layer1_vector(self, index):
        l1_vector = np.zeros(self.cmac.num_active_cells, np.int)
        mod = index % self.num_active_cells
        value = index
        for i in range(mod, self.cmac.num_active_cells):
            l1_vector[i] =  value
            value = value + 1
        for i in range(mod):
            l1_vector[i] = value
            value = value + 1
        return l1_vector

    @property
    def cmac(self):
        return self._cmac

    @cmac.setter
    def cmac(self, value):
        self._cmac = value

    @property
    def max_value(self):
        return self.get_layer1_vector(self.num_discret_values - 1).max()

    @property
    def discret_values(self):
        return self._discret_values
    @property
    def num_active_cells(self):
        return self._cmac.num_active_cells
    @property
    def num_discret_values(self):
        return self._num_discret_values
    @property
    def s_min(self):
        return self._s_min
    @property
    def s_max(self):
        return self._s_max
    @property
    def desc(self):
        return self._desc
