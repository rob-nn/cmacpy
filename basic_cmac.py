import oga_api.ml.cmac as cmac
import numpy as np
import matplotlib.pyplot as plt
import oga_api.physics.cinematic as c

class BasicCMAC(cmac.CMAC):
    def __init__(self, trajectories, pos_angles, time_frame, markers, angles, activations, output, num_iterations):
        self._num_iterations = num_iterations
        confs = []
        conf = None
        data_set = None
        for marker in markers:
            if 'xCheckedForInput' in marker and marker['xCheckedForInput'] and 'qx'in marker:
                data = c.get_vectorial_velocities(trajectories[marker['index'], 0, :], time_frame)
                conf = cmac.SignalConfiguration(data.min(), data.max(), marker['qx'],  marker['description'])
                if conf != None: confs.append(conf)        
                if data_set == None: data_set = np.reshape(data, (len(data), 1))
                else: data_set = np.concatenate((data_set,  np.reshape(data, (len(data), 1))), axis=1)
            if 'yCheckedForInput' in marker and marker['yCheckedForInput'] and 'qy'in marker:
                data = c.get_vectorial_velocities(trajectories[marker['index'], 1, :], time_frame)
                conf = cmac.SignalConfiguration(data.min(), data.max(), marker['qy'],  marker['description'])
                if conf != None: confs.append(conf)        
                if data_set == None: data_set = np.reshape(data, (len(data), 1))
                else: data_set = np.concatenate((data_set,  np.reshape(data, (len(data), 1))), axis=1)
            if 'zCheckedForInput' in marker and marker['zCheckedForInput'] and 'qz'in marker:
                data = c.get_vectorial_velocities(trajectories[marker['index'], 2, :], time_frame)
                conf = cmac.SignalConfiguration(data.min(), data.max(), marker['qz'],  marker['description'])
                if conf != None: confs.append(conf)        
                if data_set == None: data_set = np.reshape(data, (len(data), 1))
                else: data_set = np.concatenate((data_set,  np.reshape(data, (len(data), 1))), axis=1)

        super(BasicCMAC, self).__init__(confs, activations)
        if data_set == None: 
            raise ParameterInvalid('No data do process')
        if len(confs) == 0:
            raise ParameterInvalid('No input valid input sginal')

        self._data_set = data_set
        self._get_output_data(output, trajectories, pos_angles, time_frame)
        self._generate_data_for_training_and_test()

    @property
    def data_in(self):
        return self._data_in

    @property
    def data_in_test(self):
        return self._data_in_test

    @property
    def data_set(self):
        return self._data_set

    @property
    def out_data(self):
        return self._out_data

    @property
    def data_out(self):
        return self._data_out

    @property
    def data_out_test(self):
        return self._data_out_test

    def _get_output_data(self, output, trajectories, pos_angles, time_frame):
        if output['type'] == 0: #Marker
            component = 0
            if output['component'] =='x': 
                component = 0
            elif output['component'] == 'y': 
                component = 1
            else: 
                component == 2 # component == z
            self._out_data = trajectories[output['_id'], component, :]
        else: #1 Angle
            #import pdb; pdb.set_trace()
            angle = pos_angles[int(output['_id'])]
            origin = trajectories[int(angle['origin']), 0:3, :]
            component_a = trajectories[int(angle['component_a']), 0:3, :]
            component_b = trajectories[int(angle['component_b']), 0:3, :]
            if output['component'] == 'a': # angle
                self._out_data = c.get_angles(origin.T, component_a.T, component_b.T)  
            else: # v - angular velocities
                self._out_data = c.calc_angular_velocities(origin.T, component_a.T, component_b.T, time_frame)  
            #import pdb; pdb.set_trace()
                
    def _generate_data_for_training_and_test(self):
        data_in = None
        data_in_test = None
        data_out = np.array([]) 
        data_out_test = np.array([])
        
        for i in np.arange(self._data_set.shape[0]):
            if i % 2 == 0:
                if data_in == None: 
                    data_in = np.reshape(self._data_set[i,:], (1, self._data_set.shape[1]))
                else:
                    data_in = np.concatenate((data_in, np.reshape(self._data_set[i,:], (1, self._data_set.shape[1]))))
                data_out = np.append(data_out, np.array([self._out_data[i]]))
            else:
                if data_in_test == None: 
                    data_in_test = np.reshape(self._data_set[i,:], (1, self._data_set.shape[1]))
                else:
                    data_in_test = np.concatenate((data_in_test, np.reshape(self._data_set[i,:], (1, self._data_set.shape[1]))))
                data_out_test = np.append(data_out_test, np.array([self._out_data[i]]))
        self._data_in = data_in
        self._data_in_test = data_in_test
        self._data_out = data_out
        self._data_out_test = data_out_test

    def train(self): 
        if self._num_iterations < 1:
            raise ParameterInvalid('Number of iterations must be greater than 1')
        t = cmac.Train(self, self._data_in, self._data_out, 1, self._num_iterations)
        t.train()
        self.t = t

    def fire_all(self, inputs):
        result = []
        for data in inputs:
            result.append(self.fire(data))
        return np.array(result)

   
    def fire_test(self):
        return self.fire_all(self._data_in_test)


""" 
    def plot_aproximation(self, time = None):
	real = self._data_test
	aproximations = self.fire_test)
        if time == None:
            t = arange(0, real.shape[0]) * (1./315.)
        else:
            t = time
        plt.figure()
        plt.plot(self.t.E)
        
        plt.figure()
        plt.hold(True)
        p1 = plt.plot(t.tolist(), real, 'b', linewidth=4)
        p2 = plt.plot(t.tolist(), aproximation, 'r', linewidth=2)
        plt.xlabel('t (sec.)', fontsize=15)
        plt.ylabel('Angular Velocities (rads/sec.)', fontsize=15)
        plt.legend(['Human Knee', 'CMAC Prediction'])
        plt.show()

""" 
class ParameterInvalid(BaseException):
    def __init__(self, description):
        self._description = description

    @property
    def description(self):
        return self._description


