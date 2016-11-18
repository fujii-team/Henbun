from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from contextlib import contextmanager
from functools import wraps
import sys
from . import transforms
from .param import Parentable, Variable, Parameterized, Data, MinibatchData, graph_key
from ._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class Model(Parameterized):
    """
    Base class that can be a highest_parent.
    This class does basic operation,
    - assign Variable._array -> Variable._tf_array
    """
    def __init__(self, name='model'):
        """
        name is a string describing this model.
        """
        Parameterized.__init__(self)
        #self.scoped_keys.extend(['build_likelihood', 'build_prior'])
        self._name = name
        self._needs_recompile = True
        self._session = tf.Session()
        self.index = Indexer()
        # setUp the model
        self.setUp()

    @property
    def name(self):
        return self._name

    def setUp(self):
        """
        The model creation should be done here in the inheriting method.
        """
        pass

    def initialize(self):
        # TODO should support tensorflow 0.12
        # self._session.run(tf.variables_initializer(self.parameters))
        self._session.run(self.initialize_ops)
        self.finalize()

    def validate(self):
        """
        Make some validation of this model.
        """
        # --- check all the LOCAL variables are fed ---
        for p in self.get_variables(graph_key.LOCAL):
            if p.tensor is None:
                raise ValueError('local variable '+p.long_name+' is not fed.')

        # --- check the data size ---
        minibatch_data = [d for d in self.get_variables(graph_key.DATA)
                                if isinstance(d, MinibatchData)]
        if len(minibatch_data) > 1:
            for d in minibatch_data:
                if d.data_size != minibatch_data[0].data_size:
                    raise ValueError('Minibatch data'+d.long_name+' is not the same size.')
        if len(minibatch_data) > 0:
            data_size = minibatch_data[0].data_size
            if self.index.data_size is None or self.index.data_size != data_size:
                self.index.setUp(data_size)

    def test_feed_dict(self, minibatch_size=None):
        """
        Return feed_dict for test data
        """
        return self.get_feed_dict(self.index.test_index(minibatch_size))

    def run(self, tensor, feed_dict=[]):
        return self._session.run(tensor, feed_dict=feed_dict)

class Indexer(object):
    def __init__(self):
        self.data_size = None
        self.test_frac = 0.1

    def setUp(self, data_size):
        """
        data_size: size of the entire data
        test_frac: fraction of test data from the entire data
        """
        self.data_size = data_size
        self.test_size  = int(np.floor(self.data_size*self.test_frac))
        self.train_size = data_size - self.test_size
        index = np.array(range(self.data_size))
        np.random.shuffle(index)
        self._train_index = index[:self.train_size]
        self._test_index  = index[self.train_size:]

    def train_index(self, minibatch_size):
        """ Return random index from training data """
        return self._train_index[np.random.randint(0, self.train_size, minibatch_size)]

    def test_index(self, minibatch_size):
        """ Return random index from test data """
        return self._test_index[np.random.randint(0, self.test_size, minibatch_size)]

class AutoOptimize(object):
    """
    This decorator class is designed to wrap the likelihood method in models
    in order to enable the optimization or simple evaluation.

    The typical usage is
    >>> @AutoOptimize()
    >>> def likelihood(self):
    >>>     logp = ... some calculation ...
    >>>     return logp

    >>> likelihood.eval()
    returns the objective value.

    >>> likelihood.optimize(collection)
    optimizes and update parameters.
    """
    def __init__(self):
        pass

    def __call__(self, method):
        @wraps(method)
        def runnable(instance):
            optimizer_name = '_'+method.__name__+'_AF_optimizer'
            if hasattr(instance, optimizer_name):
                # the method has been compiled already.
                optimizer = getattr(instance, optimizer_name)
            else:
                # the method deeds to be compiled
                optimizer = Optimizer(instance, method)
                setattr(instance, optimizer_name, optimizer)
            return optimizer

        return runnable

class Optimizer(object):
    """
    Optimizer object that will be handled by AutoOptimizer class.
    """
    def __init__(self, model_instance, likelihood_method):
        """
        model_instance: instance of model class.
        likelihood_method: method to be optimized.
        """
        self.model = model_instance
        self.likelihood_method = likelihood_method
        # Op for the likelihood evaluation
        self.method_op = None
        # Op for the optimzation
        self.optimize_op = None

    def compile(self, optimizer = tf.train.AdamOptimizer(),
                collection=graph_key.VARIABLES):
        """
        Create self.method_op and self.optimize_op.
        """
        print('compiling...')
        var_list = self.model.get_tf_variables(collection)
        with self.model.tf_mode():
            self.method_op = self.likelihood_method(self.model)
            self.optimize_op = optimizer.minimize(tf.neg(self.method_op),
                                                            var_list=var_list)
        # manual initialization.
        self.model.initialize()
        # initialize un-initialized variable
        self.model._session.run(tf.initialize_variables(
            [v for v in tf.all_variables() if not
             self.model._session.run(tf.is_variable_initialized(v))]))
        # make validation
        self.model.validate()

    def feed_dict(self, minibatch_size=None, training=True):
        if minibatch_size is None:
            return self.model.get_feed_dict(None)
        elif training:
            return self.model.get_feed_dict(
                                self.model.index.train_index(minibatch_size))
        else:# test
            return self.model.get_feed_dict(
                                self.model.index.test_index(minibatch_size))

    def run(self, minibatch_size=None, training=True):
        return self.model._session.run(self.method_op,
                    feed_dict=self.feed_dict(minibatch_size, training))

    def optimize(self, maxiter=1, minibatch_size=None, callback=None):
        """
        target: method to be optimized.
        trainer: tf.train object.
        """
        iteration = 0
        while iteration < maxiter:
            self.model._session.run(self.optimize_op,
                        feed_dict=self.feed_dict(minibatch_size))
            if callback is not None:
                callback()
            iteration += 1
