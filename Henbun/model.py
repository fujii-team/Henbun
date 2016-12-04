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

    All the models must inherite this class.
    Variables, Parameterized should be defined in setUp method.

    The typical usage is to define a method to be optimized with decrated by
    AutoOptimize,

    >>> class SquareModel(hb.model.Model):
    >>>     def setUp(self):
    >>>         self.p = hb.param.Variable([2,3])
    >>>
    >>>     @hb.model.AutoOptimize()
    >>>     def likelihood(self):
    >>>         return -tf.reduce_sum(tf.square(self.p))
    >>> m = SquareModel()
    where
    >>> m.likelihood().compile()
    makes a tensorflow graph based that evaluates and optimize likelihood()
    method.

    >>> m.likelihood().run()
    returns the value of m.likelihood() with the current parameter values.

    >>> m.likelihood().optimize(maxiter = 1000)
    optimizes global parameters in the graph that maximizes m.likelihood().


    This class also does basic operations such as
    >>> m.initialize()
    assignes the value to parameters.

    """
    def __init__(self, name='model'):
        """
        name is a string describing this model.
        """
        Parameterized.__init__(self)
        # name of the model
        self._name = name
        # tf.Session to run the graph
        self._session = tf.Session()
        # Index that will be used for minibatching.
        self._index = Indexer()
        # setUp the model
        self.setUp()

        # TODO some tricks to avoid recompilation
        #self._needs_recompile = True

        self._saver = self._get_saver()

    def _get_saver(self):
        """ prepare saver """
        # --- setup savers.---
        var_dict = {v.long_name: v._tensor for v in self.get_variables()
                if v.collections not in graph_key.not_parameters}
        if len(var_dict.keys()) > 0:
            return tf.train.Saver(var_dict)
        else:
            return None

    @property
    def name(self):
        return self._name

    def setUp(self):
        """
        Definition of parameters should be done in this method.
        """
        pass

    def initialize(self):
        """
        Run the assignment ops that is gathered by self.initialize_op
        """
        # TODO should support tensorflow 0.12
        # self._session.run(tf.variables_initializer(self.parameters))
        self._session.run(self.initialize_ops)
        self.finalize()

    def save(self, save_path = None, global_step = None):
        """
        Returns: path of the saved-file.
        """
        if save_path is None:
            save_path = self.name + '.ckpt'
        #
        assert self._saver is not None
        # do initialization for the case variables are not initialized.
        self.initialize()
        # save
        return self._saver.save(self._session, save_path, global_step)

    def restore(self, save_path=None):
        """
        Restore the parameter from file.
        """
        if save_path is None:
            save_path = self.name + '.ckpt'
        self._saver.restore(self._session, save_path)
        # raise down the initialized flag
        [v.finalize() for v in self.get_variables()]

    def run(self, tensor, feed_dict=None):
        """
        Shortcut for self._session.run(operation)
        args:
        - tensor: tensor or operation to be evaluated.
        - feed_dict: feed dictionary to run the tensor.
                    If this model do not have MinibatchData, feed_dict=None
                    can be used to feed all the data.
        """
        if feed_dict is None:
            feed_dict = self.get_feed_dict()
        return self._session.run(tensor, feed_dict=feed_dict)

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
            if self._index.data_size is None or self._index.data_size != data_size:
                self._index.setUp(data_size)

    def test_feed_dict(self, minibatch_size=None):
        """
        Return feed_dict for test data
        """
        return self.get_feed_dict(self._index.test_index(minibatch_size))


class Indexer(object):
    """
    A simple class that handles minibatching.
    """
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
    This decorator class is designed to wrap methods in models
    in order to enable the optimization or simple evaluation.

    The typical usage is
    >>> @AutoOptimize()
    >>> def likelihood(self):
    >>>     logp = ... some calculation ...
    >>>     return logp

    >>> likelihood().run()
    returns the objective value.

    >>> likelihood().optimize(collection)
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
    An object handled by AutoOptimizer class.
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
                global_step=None, collection=graph_key.VARIABLES,
                gate_gradients=1, aggregation_method=None,
                colocate_gradients_with_ops=False, grad_loss=None):
        """
        Create self.method_op and self.optimize_op.
        args:
        - optimzer: instance of tf.train.optimizer.
        - collection: variable collection that will be optimized.
        - global_step: If want to decrease learning rate, global_step can be
                        passed.
        """
        print('compiling...')
        var_list = self.model.get_tf_variables(collection)
        with self.model.tf_mode():
            self.method_op = self.likelihood_method(self.model)
            self.optimize_op = optimizer.minimize(tf.neg(self.method_op),
                    global_step=global_step, var_list=var_list,
                    gate_gradients=gate_gradients, aggregation_method=aggregation_method,
                    colocate_gradients_with_ops=colocate_gradients_with_ops,
                    grad_loss=grad_loss)
        # manual initialization.
        self.model.initialize()
        # initialize un-initialized variable
        self.model._session.run(tf.initialize_variables(
            [v for v in tf.all_variables() if not
             self.model._session.run(tf.is_variable_initialized(v))]))
        # make validation
        self.model.validate()

    def feed_dict(self, minibatch_size=None, training=True):
        """
        A method to deal with feed_dict
        """
        if minibatch_size is None:
            return self.model.get_feed_dict(None)
        elif training:
            return self.model.get_feed_dict(
                                self.model._index.train_index(minibatch_size))
        else:# test
            return self.model.get_feed_dict(
                                self.model._index.test_index(minibatch_size))

    def run(self, minibatch_size=None, training=True):
        """
        Method to evaluate the method with the current parameters.
        """
        try:
            return self.model._session.run(self.method_op,
                        feed_dict=self.feed_dict(minibatch_size, training))
        except KeyboardInterrupt:
            raise KeyboardInterrupt

    def optimize(self, maxiter=1, minibatch_size=None):
        """
        Method to optimize the self.method.
        args:
        - maxiter: number of iteration.
        - minibatch_size: size of Minibatching.
        """
        iteration = 0
        while iteration < maxiter:
            try:
                self.model._session.run(self.optimize_op,
                            feed_dict=self.feed_dict(minibatch_size))
                iteration += 1
            except KeyboardInterrupt:
                raise KeyboardInterrupt
