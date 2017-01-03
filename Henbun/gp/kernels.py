# Copyright 2016 Keisuke Fujii
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

# We learned a lot from GPflow https://github.com/GPflow/GPflow/
# Thanks GPflow!

import tensorflow as tf
import numpy as np
from .. import transforms
from ..param import Variable, Parameterized, graph_key
from ..variationals import Variational
from ..tf_wraps import eye
from .._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class Kern(Parameterized):
    """
    The basic kernel class.
    """
    def __init__(self):
        Parameterized.__init__(self)
        self.scoped_keys.extend(['K','Kdiag'])

class UnitStationary(Kern):
    def __init__(self, lengthscales=np.ones(1),
            n_batch = None, collections=[graph_key.VARIABLES]):
        """
        lengthscales is [scalor or 1d-np.array]
        """
        Kern.__init__(self)
        if isinstance(lengthscales, np.ndarray):
            self.lengthscales = Variable(lengthscales.shape,
                                    transform=transforms.positive, collections=collections)
            # set initial values
            self.lengthscales = lengthscales
        elif isinstance(lengthscales, (Variable, Variational)):
            self.lengthscales = lengthscales
        else:
            raise TypeError
        self.scoped_keys.extend(['square_dist', 'euclid_dist', 'Cholesky'])

    def square_dist(self, X, X2=None):
        """
        Returns the square distance between X and X2.
        X, X2 is 2d- or 3d-tensor.

        If X is (and X2 should have the same dimension) 2dimensional,
        each dimension is considered as [n,d] and [n2,d]
         - n: number of data point
         - d: dimension of each data
        This method returns [n,n2] sized kernel value.
        If X2 is None, then X2 = X is assumed.

        If X is 3-dimensional,
        each dimension is considered as [N,n,d] (and [N,n2,d] for X2)
         - N: batch number.
        Returns: [N,n,n2] sized kernel value.
        """
        Xeff = X/self.lengthscales # [n,d]  or [N,n,d]
        Xs = tf.reduce_sum(tf.square(Xeff), -1) # [n] or [N,n]
        if X2 is None:
            # batched case : [N,n,d]@[N,d,n]->[N,n,n]
            # non-batch case:[n,d]@[d,n]->[n,n]->[n,n]
            return -2*tf.matmul(Xeff, Xeff, transpose_b=True) + \
                        tf.expand_dims(Xs, -1) + tf.expand_dims(Xs, -2)
        else:
            X2eff = X2/self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2eff), -1)
            # batched case : [N,n,d]@[N,d,n2]->[N,n,n2]
            # non-batch case:[n,d]@[d,n]->[n,n]->[n,n]
            return -2*tf.matmul(Xeff, X2eff, transpose_b=True) + \
                        tf.expand_dims(Xs, -1) + tf.expand_dims(X2s, -2)

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X):
        return tf.ones(tf.shape(X)[:-1], dtype=float_type)

    def Cholesky(self, X):
        """
        Cholesky decomposition of K(X).
        If X is sized [n,d], this returns [n,n].
        If X is sized [N,n,d], this returns [N,n,n] where each [n,n] matrix
        is lower triangular.
        """
        jitter = eye(tf.shape(X)[-2])*settings.numerics.jitter_level
        return tf.cholesky(self.K(X)+jitter)

class UnitRBF(UnitStationary):
    """
    The radial basis function (RBF) or squared exponential kernel
                     (x-x2)^2
    K(x,x2) = exp(- ----------)
                     2 * l^2
    """
    def K(self, X, X2=None):
        return tf.exp(-self.square_dist(X, X2)/2)

class UnitCsymRBF(UnitStationary):
    """
    The squared exponential kernel in cylindrically symmetric space.
                     (x-x2)^2            (x+x2)^2
    K(x,x2) = exp(- ----------) + exp(- ----------)
                     2 * l^2             2 * l^2
    The second term indicates the correlation between the opsite
    side of the point against the axis x=0.
    """
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return tf.exp(-self.square_dist(X,  X2)/2)\
             + tf.exp(-self.square_dist(X, -X2)/2)

    def Kdiag(self, X):
        Xeff = X/self.lengthscales # [n,d]  or [N,n,d]
        Xs = tf.reduce_sum(tf.square(Xeff), -1) # [n] or [N,n]
        return tf.ones_like(Xs, dtype=float_type) + tf.exp(-2*Xs)
