import tensorflow as tf
import numpy as np
import traceback
from .param import Variable, Parameterized, graph_key
from .tf_wraps import clip
from ._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class MatBias(Parameterized):
    def __init__(self, nodes, n_layers=[],
                    mean=0.0, stddev=1.0,
                    variable = tf.Variable,
                    collections = [graph_key.VARIABLES]):
        """
        A simple class that handles matrix and baias pair, w*x + b
        input:
        - nodes: 2-integer-element list or tuples indicating number of input and
               output for the matrix
        """
        assert(len(nodes)==2)
        Parameterized.__init__(self)
        # --- define matrices and biases ---
        self.w = variable(shape=[nodes[0], nodes[1]], n_layers=n_layers,
                            mean=mean, stddev=stddev,
                            collections=collections)
        self.b = variable(shape=[1,nodes[1]], n_layers=n_layers,
                            mean=mean, stddev=stddev,
                            collections=collections)

    def __call__(self, x):
        return clip(tf.matmul(x, self.w) + self.b)

class NeuralNet(Parameterized):
    def __init__(self, nodes, n_layers = [],
                        mean=0.0, stddev=1.0,
                        variable_types = Variable,
                        neuron_types = tf.sigmoid,
                        collections = [graph_key.VARIABLES]):
        """
        nodes: list of nodes num of the neural net.
        n_layers: number of layers.
        variable typs: single or list of Variable object, one of
                        [tf.Variable or Variational]
        types: single or list of n.n. object.
        name: name of this neural net
        """
        Parameterized.__init__(self)
        self.nodes = nodes
        # --- variable types ----
        if not isinstance(variable_types, list):
            variable_types = [variable_types for _ in range(len(nodes)-1)]
        else:
            variable_types = variable_types
        # --- neuron types ----
        if not isinstance(neuron_types, list):
            self.neuron_types = [neuron_types for _ in range(len(nodes)-2)]
        else:
            self.neuron_types = neuron_types
        # --- define matrices and biases ---
        self._matbias_list = []
        for i in range(len(nodes)-1):
            matbias = MatBias(nodes=[nodes[i], nodes[i+1]],
                            n_layers=n_layers,
                            mean=mean, stddev=stddev,
                            variable = variable_types[i],
                            collections=collections)
            self._matbias_list.append(matbias)
            key = 'matbias' + str(i)
            name_matbias = self.name + str('.matbias')
            setattr(self, key, matbias)

    def __call__(self, x):
        """
        x: tf.tensor
        Returns Op for this neural net.
        This method should be executed in tf_mode
        """
        y = x
        for i in range(len(self.nodes)-2):
            typ = self.neuron_types[i]
            name_nn = None if self.name is None else self.name + str('.nn')+str(i)
            y = typ(self._matbias_list[i](y), name=name_nn)
        return self._matbias_list[-1](y)

    def __getitem__(self, i):
        return self._matbias_list[i]
