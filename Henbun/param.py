# Copyright 2016 Keisuke Fujii
#
# We studied a lot from GPflow https://github.com/GPflow/GPflow.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from contextlib import contextmanager
from functools import wraps, reduce
from . import transforms
from .scoping import NameScoped
from ._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class _GraphKey(object):
    def __init__(self):
        self.VARIABLES = tf.GraphKeys.VARIABLES
        self.LOCAL = 'LOCAL'
        self.DATA = 'DATA'

    @property
    def not_parameters(self):
        return [self.LOCAL, self.DATA]

graph_key = _GraphKey()

class Parentable(object):
    """
    A very simple class for objects in a tree, where each node contains a
    reference to '_parent'.
    This class can figure out its own name (by seeing what it's called by the
    _parent's __dict__) and also recurse up to the highest_parent.

    The code of this class is quated from
    https://github.com/GPflow/GPflow/blob/master/GPflow/param.py
    """
    def __init__(self):
        self._parent = None

    @property
    def highest_parent(self):
        """A reference to the top of the tree, usually a Model instance"""
        if self._parent is None:
            return self
        else:
            return self._parent.highest_parent

    @property
    def name(self):
        """An automatically generated name, given by the reference of the _parent to this instance"""
        if self._parent is None:
            return 'unnamed'
        if isinstance(self._parent, ParamList):
            return 'item%i' % self._parent._list.index(self)
        matches = [key for key, value in self._parent.__dict__.items()
                   if value is self]
        if len(matches) == 0:
            raise ValueError("mis-specified parent. This Param's\
                             _parent does not contain a reference to it.")
        if len(matches) > 1:
            raise ValueError("This Param appears to be doubly\
                             referenced by a parent")
        return matches[0]

    @property
    def long_name(self):
        """
        This is a unique identifier for a param object within a structure, made
        by concatenating the names through the tree.
        """
        if self._parent is None:
            return self.name
        return self._parent.long_name + '.' + self.name

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_parent')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._parent = None

class Variable(Parentable):
    """
    Abstract class for LocalVariable, GlobalVariable, GlobalData,
    LocalVariational, GlobalVariationals
    """
    def __init__(self, shape, n_layers=[], n_batch=None,
        transform=transforms.Identity(), collections=[graph_key.VARIABLES]):
        """
        shape: list or tuples indicating the shape of this parameters.
                In the LOCAL case, the right most axis is MinibatchSize.
                This axis can be None. In this case, we do not validate the shape.
        n_layers: List of integers indicating number of layers.
        n_batches: Integer representing number of batches. It can be None.
                In Local case, the batch_size is automatically determined if
                None is given. If a certain value is specified, then Local and
                Global variables behave same.
        transform: one of transforms.
        collections: List of strings or 'LOCAL'.
        If 'LOCAL' is assigned, this object does not create tf.Variable instead
        feed(x) method should be used.
        """
        Parentable.__init__(self)
        if isinstance(shape, int):
            shape = [shape]
        self.transform = transform
        self.collections = collections
        self.n_batch = n_batch
        self.shape = list(shape)
        self.n_layers = list(n_layers) # number of layers.
        self._assigned = True
        if self.collections in graph_key.not_parameters:
            # In LOCAL case, tensor will be feeded by self.feed method.
            # In DATA case, PlaceHolder class should take care self._tensor
            self._tensor = None # the value of this param. It will be sized [shape[:-1],N]
        else:
            if self.n_batch is None:
                _shape = list(n_layers) + list(shape)
            else:
                _shape = list(n_layers) + list(shape) + [self.n_batch]
            self._tensor = tf.Variable(tf.truncated_normal(_shape, dtype=float_type),
                    dtype=float_type, collections=collections)
            self._initialize_op = tf.initialize_variables([self._tensor])

    def tensor(self):
        if self._tensor is None:
            return None
        return self.transform.tf_forward(self._tensor)

    def get_tf_variables(self, collection):
        if collection in self.collections:
            return [self._tensor]
        else:
            return []

    def get_variables(self, collection):
        if collection in self.collections:
            return [self]
        else:
            return []

    def assign(self, value):
        """
        Assign value for self._tensor.
        The initialize_ops is updated and _assigned flag raised.
        """
        self._initialize_op = self._tensor.assign(self.transform.backward(value))
        self._assigned = True

    @property
    def initialize_ops(self):
        if self.collections not in graph_key.not_parameters and self._assigned:
            return [self._initialize_op]
        else:
            return []

    def finalize(self):
        """
        Remove a initialize_op so that the next initialization does not change
        the parameters.
        """
        self._assigned = False

    @property
    def value(self):
        assert(hasattr(self.highest_parent, '_session'))
        return self.highest_parent._session.run(self.tensor())

    @property
    def feed_size(self):
        if self.collections is graph_key.LOCAL:
            return reduce(np.multiply, self.shape, 1)
        else:
            return 0

    def feed(self, x):
        """
        Feed values to this local variable.
        x: tensor sized [*self._shape, N] where N is the minibatch size.
        """
        if self.collections is graph_key.LOCAL:
            # check if the shape is the same
            if hasattr(x, "get_shape"):
                shape = x.get_shape()
                if self.n_batch is not None and shape[-1] is not None:
                    assert(shape[-1]==self.n_batch)
            shape2 = self.n_layers + self.shape + [tf.shape(x)[-1],]
            self._tensor = tf.reshape(x, shape2)

    def get_feed_dict(self, minibatch_index):
        """
        This method should be implemented in the child class
        """
        feed_dict = {}
        if self.collections is graph_key.DATA:
            raise NotImplementedError
        else:
            return feed_dict

class Parameterized(Parentable):
    """
    Object that holds parameters.

    This object is designed to be part of a tree, with Variables at the leaves.
    We can then recurse down the tree to find all the Variables (leaves),
    or recurse up the tree (using highest_parent)
    from the leaves to the root.

    A useful application of such a recursion is 'tf_mode', where the parameters
    appear as their _tf_array variables. This allows us to build models on
    those parameters. During _tf_mode, the __getattribute__ method is
    overwritten to return tf arrays in place of parameters (and data).
    Another recursive function is build_prior which sums the log-prior from all
    of the tree's parameters (whilst in tf_mode!).
    *Scoping*
    Parameterized classes can define functions that operate on tf variables. To
    wrap those functions in tensorflow scopes, the names of the scoped
    fucntions are stored in self.scoped_keys (a list of strings). Those
    functions are then called inside a tensorflow scope.
    """
    def __init__(self):
        Parentable.__init__(self)
        self._tf_mode = False
        self.scoped_keys = []

    def __getattribute__(self, key):
        """
        Here, we overwrite the getattribute method.
        If tf mode is off, this does nothing.
        If tf mode is on, all child parameters will appear as their tf
        representations, and all functions that are designated in 'scoped_keys'
        will have aname scope applied.
        """
        o = object.__getattribute__(self, key)

        # if _tf_mode is False, or there is no _tf_mode, just return the object as normal.
        try:
            if not object.__getattribute__(self, '_tf_mode'):
                return o
        except AttributeError:
            return o

        # return _parent as it is.
        if key is '_parent':
            return o

        # In tf_mode, if the object is a Parameterized and it has tensor attribute,
        # then return its tensor
        if isinstance(o, (Parameterized, Variable)) and hasattr(o, 'tensor'):
            return o.tensor()

        # in tf_mode, wrap functions is a scope
        elif key in object.__getattribute__(self, 'scoped_keys'):
            return NameScoped(self.long_name + '.' + key)(o)

        # finally, just return the object
        return o

    def __setattr__(self, key, value):
        """
        We overwrite __setattr__ to Variable that feed to its tensor.
        """
        # If we already have an atribute with that key, decide what to do:
        if key in self.__dict__.keys():
            p = object.__getattribute__(self, key)
            # In tf_mode, the setattribute for Local or Variational parameters
            # are replaced by feed
            try:
                if object.__getattribute__(self, '_tf_mode'):
                    if isinstance(p, (Variable, Parameterized)):
                        p.feed(value)
                        return
            except:
                pass
            # if the existing attribute is a parameter, and the value is an
            # array (or float, int), then set the _array of that parameter
            if isinstance(p, Variable) and isinstance(p._tensor, tf.Variable):
                if isinstance(value, (float, int)):
                    value = np.array([value], dtype=np_float_type)
                if isinstance(value, np.ndarray):
                    p.assign(value)
                    return
            # if the existing attribute is a Param (or Parameterized), and the
            # new attribute is too, replace the attribute and set the model to
            # recompile if necessary.
            if isinstance(p, (Variable, Parameterized)) \
                    and isinstance(value, (Variable, Parameterized)):
                p._parent = None  # unlink the old Parameter from this tree
                if hasattr(self, '_needs_recompile'):
                    self.highest_parent._needs_recompile = True

        # use the standard setattr
        object.__setattr__(self, key, value)

        # make sure a new child node knows this is the _parent:
        if isinstance(value, Parentable) and key is not '_parent':
            value._parent = self

    @contextmanager
    def tf_mode(self):
        """
        A context for building models.
        Correct usage is:
        with m.tf_mode:
            # do tf stuff, like
            m.build_likelihood()
            m.build_prior()
        with this context engaged, any Param objects which are children of this
        class will appear as their tf-variables. Example
        >>> m = Parameterized()
        >>> m.foo = Param(1.0)
        >>> m.make_tf_array(tt.dvector())
        >>> print m.foo
        foo
        [ 1.]
        >>> with m.tf_mode():
        >>>     print m.foo
        Reshape{1}.0
        The idea is that in tf_mode, we can easily get references to the
        tf representation of parameters in order to construct tf
        objective functions.
        """
        self._begin_tf_mode()
        yield
        self._end_tf_mode()

    def _begin_tf_mode(self):
        [child._begin_tf_mode() for child in self.sorted_variables
         if isinstance(child, Parameterized)]
        self._tf_mode = True

    def _end_tf_mode(self):
        [child._end_tf_mode() for child in self.sorted_variables
         if isinstance(child, Parameterized)]
        self._tf_mode = False

    @property
    def sorted_variables(self):
        """
        Return a list of all the child variables, sorted by name.
        This makes sure they're always in the same order.
        This method works also in tf_mode
        """
        variables = [child for key, child in self.__dict__.items()
                  if isinstance(child, (Variable, Parameterized))
                  and key is not '_parent']
        return sorted(variables, key=lambda x: x.name)

    def get_tf_variables(self, collection=graph_key.VARIABLES):
        """
        Return a list of all the child parameters that should be optimized.
        """
        params = []
        for p in self.sorted_variables:
            params += p.get_tf_variables(collection)
        return params

    def get_variables(self, collection):
        params = []
        for p in self.sorted_variables:
            params += p.get_variables(collection)
        return params

    @property
    def initialize_ops(self):
        """
        Return a list of all the child parameters that should be optimized.
        """
        params = []
        for p in self.sorted_variables:
            params += p.initialize_ops
        return params

    def finalize(self):
        """
        Remove a initialize_op so that the next initialization does not change
        the parameters.
        """
        for p in self.sorted_variables:
            p.finalize()

    @property
    def feed_size(self):
        """
        Returns the total feed size of all the child LocalVariables.
        The last axis of each feed_size must be -1,
        The summation is taken along the second last axis.
        The other axis for all the variables should be the same.
        """
        return np.sum([p.feed_size for p in self.get_variables(graph_key.LOCAL)],
                                                                    dtype=int)

    def feed(self, x):
        """
        Feed tensor x into all the child LocalVariable
        """
        # --- assertion ---
        assert len(self.get_variables(graph_key.LOCAL))>0
        n_layers=self.get_variables(graph_key.LOCAL)[0].n_layers
        for p in self.get_variables(graph_key.LOCAL):
            assert len(p.n_layers) == len(n_layers)
            assert all([n==n0 for n,n0 in zip(p.n_layers, n_layers)]), '''
            n_layers of all the LOCAL variables should be same for using this method. \n
            If not, feed separately by hand instead.'''
        # offset
        begin = np.zeros(len(n_layers) + 2)
        size = -np.ones(len(n_layers) + 2)
        for p in self.get_variables(graph_key.LOCAL):
            size[-2] = p.feed_size
#            p.feed(tf.slice(x, begin, size))
            p.feed(tf.slice(x, tf.convert_to_tensor(begin, dtype=tf.int32),
                               tf.convert_to_tensor(size,  dtype=tf.int32)))
            begin[-2] += p.feed_size

    def get_feed_dict(self, minibatch_index):
        feed_dict = {}
        for p in self.sorted_variables:
            feed_dict.update(p.get_feed_dict(minibatch_index))
        return feed_dict

    def KL(self):
        """
        Returns the sum of KL values for all the childs.
        """
        KL_list = [p.KL() for p in self.sorted_variables
                    if hasattr(p, 'KL')]
        if len(KL_list) == 0: # for the zero-list case
            return np.zeros([], dtype=np_float_type)
        else:
            return reduce(tf.add, KL_list)


class ParamList(Parameterized):
    """
    A list of parameters.
    This allows us to store parameters in a list whilst making them 'visible'
    to the GPflow machinery. The correct usage is
    >>> my_list = GPflow.param.ParamList([Param1, Param2])
    You can then iterate through the list. For example, to compute the sum:
    >>> my_sum = reduce(tf.add, my_list)
    or the sum of the squares:
    >>> rmse = tf.sqrt(reduce(tf.add, map(tf.square, my_list)))
    You can append things:
    >>> my_list.append(GPflow.kernels.RBF(1))
    but only if the are Parameters (or Parameterized objects). You can set the
    value of Parameters in the list:
    >>> my_list = GPflow.param.ParamList([GPflow.param.Param(2)])
    >>> print my_list
    unnamed.item0 transform:(none) prior:None
    [ 2.]
    >>> my_list[0] = 12
    >>> print my_list
    unnamed.item0 transform:(none) prior:None
    [ 12.]
    But you can't change elements of the list by assignment:
    >>> my_list = GPflow.param.ParamList([GPflow.param.Param(2)])
    >>> new_param = GPflow.param.Param(4)
    >>> my_list[0] = new_param # raises exception
    """

    def __init__(self, list_of_params=[]):
        Parameterized.__init__(self)
        for item in list_of_params:
            assert isinstance(item, (Variable, Parameterized))
            item._parent = self
        self._list = list_of_params

    @property
    def sorted_variables(self):
        return self._list

    def __getitem__(self, key):
        """
        If tf mode is off, this simply returns the corresponding Param .
        If tf mode is on, all items will appear as their tf
        representations.
        """
        o = self.sorted_variables[key]
        if isinstance(o, Variable) and self._tf_mode:
            return o.tensor()
        return o

    def append(self, item):
        assert isinstance(item, (Variable, Parameterized)), \
            "this object is for containing parameters"
        item._parent = self
        self.sorted_variables.append(item)

    def __setitem__(self, key, value):
        """
        It's not possible to assign to things in the list, but it is possible
        to set their values by assignment.
        """
        p = self.sorted_variables[key]
        if isinstance(value, np.ndarray):
            p._initialize_op = p._tensor.assign(value)
            return  # don't call object.setattr or set the _parent value
        elif isinstance(value, (float, int)):
            p._initialize_op = p._tensor.assign(np.array([value], dtype=np_float_type))
            return
        else:
            raise TypeError

class Data(Variable):
    """
    Class for feeding data into Graph.
    """
    def __init__(self, data):
        # call initializer
        Variable.__init__(self, data.shape, n_layers=[], n_batch=None,
                                                collections=graph_key.DATA)
        # prepare placeholder
        shape = list(self.n_layers) + list(self.shape)
        self._tensor = tf.placeholder(shape=shape, dtype=float_type)
        self.data = data

    def get_feed_dict(self, minibatch_index):
        """
        This method should be implemented in the child class
        """
        return {self._tensor: self.data}

class MinibatchData(Variable):
    """
    Class for feeding minibatch-data into Graph.
    """
    def __init__(self, data, n_batch=None):
        # call initializer
        Variable.__init__(self, data.shape[:-1], n_layers=[], n_batch=n_batch,
                                                collections=graph_key.DATA)
        shape = list(self.n_layers) + list(self.shape) + [n_batch]
        self._tensor = tf.placeholder(shape=shape, dtype=float_type)
        self.data = data

    @property
    def data_size(self):
        return self.data.shape[-1]

    def get_feed_dict(self, minibatch_index):
        """
        This method should be implemented in the child class
        """
        return {self._tensor: self.data[...,minibatch_index]}
