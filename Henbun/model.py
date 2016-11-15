from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from contextlib import contextmanager
from functools import wraps
import sys
from . import transforms
from .param import Parentable, Variable, Parameterized, graph_key
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

    def likelihood(self):
        """
        If there is likelihood, it should be implemented here.
        """
        return tf.zeros([],dtype=float_type)

    def build_elbo(self):
        """
        Generate tf function to calculate ELBO.
        """
        # TODO Likelihood and local.KL should be scaled to take into account
        # the minibatch operation
        #scale = self._local_manager.minibatch_frac
        scale = tf.Variable(1.0, dtype=float_type)
        lik = tf.zeros([],dtype=float_type)
        lik += self.KL(variable_types.locals) * scale
        lik += self.KL(variable_types.globals)
        return lik / tf.cast(self._tf_num_samples, dtype=float_type)

    def initialize(self):
        # TODO should support tensorflow 0.12
        # self._session.run(tf.variables_initializer(self.parameters))
        self._session.run(self.initialize_ops)
        self.finalize()

    def optimize(self, target='', maxiter=1000,
                        collections=graph_key.VARIABLES, callback=None):
        pass
'''
    def _assign_tf(self, vtypes=variable_types.free_parameters):
        """
        - assign values Variable._array -> Variable._tf_array
        """
        free_states = self.get_tensor_dict(vtypes)
        assign_op = [key.assign(item) for key, item in free_states.items()]
        self._session.run(assign_op)

    def _assign_np(self, vtypes=variable_types.free_parameters):
        """
        - assign values Variable._tf_array -> Variable._array
        """
        free_states = self.get_tensor_dict(vtypes)
        for key, item in free_states.items():
            item[...] = self._session.run(key)

    def _compile(self):
        """
        compile the tensorflow function "self._objective"
        """
        with self.tf_mode():
            self._draw_samples(self._tf_num_samples)
            f = self.build_elbo()
        self._minusF = tf.neg(f, name='objective')

        self._opt_steps = {}
        assign_ops = []
        for vtype in variable_types.free_parameters:
            free_states = self.get_tensor_dict(vtype)
            assign_ops += [key.assign(item) for key, item in free_states.items()]
            # if there is no 'local' variables, then opt_step evaluation fails.
            try:
                self._opt_steps[vtype] = self.trainer[vtype].minimize(
                                        self._minusF,
                                        var_list=list(free_states.keys())),
            except ValueError:
                pass
        # initialize
        init = tf.initialize_all_variables()
        self._session.run(init)

        # build tensorflow functions for computing the likelihood
        if settings.verbosity.tf_compile_verb:
            print("compiling tensorflow function...")
        sys.stdout.flush()

        if settings.verbosity.tf_compile_verb:
            print("done")
        sys.stdout.flush()
        self._needs_recompile = False

    def optimize(self, maxiter, callback=None, num_samples=20,
                                            vtype=variable_types.global_param):
        if self._needs_recompile:
            self._compile()
        """
        Optimize
        """
        # Make iterations.
        feed_dict = self.get_tensor_dict(variable_types.fixed_values)
        feed_dict[self._tf_num_samples] = num_samples
        try:
            iteration = 0
            while iteration < maxiter:
                self._session.run(self._opt_steps[vtype], feed_dict=feed_dict)
                if callback is not None:
                    callback(self._session.run(self._minusF, feed_dict))
                iteration += 1

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting model\
                  with most recent state.")
            self._assign_np(vtype)
            return None
        # set the result
        self._assign_np(vtype)


    def __setattr__(self, key, value):
        """
        Overload __setattr__ to raise recompilation flag if necessary
        """
        Parameterized.__setattr__(self, key, value)
        if key is 'trainer':
            self._needs_recompile = True
'''
