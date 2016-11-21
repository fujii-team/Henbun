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
from ..tf_wraps import eye
from .._settings import settings
from ..param import Variable, Parameterized, graph_key
from ..variationals import Variational
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
                                    n_batch=n_batch,
                                    transform=transforms.positive, collections=collections)
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

        If X is (and X2 should be the same dim.) 2dimensional,
        each dimension is considered as [n,d] and [n2,d]
         - n: number of data point
         - d: dimension of each data
        This method returns [n,n2] sized kernel value.
        If X2 is None, then X2 = X is assumed.

        If X is 3-dimensional,
        each dimension is considered as [n,d,N] (and [n2,d,N] for X2)
         - N: batch number.
        Returns: [n,n2,N] sized kernel value.
        """
        # match lengthscales in batched case
        def fn1(): return self.lengthscales
        def fn2(): return tf.expand_dims(self.lengthscales, -1)
        l = tf.cond(tf.equal(tf.rank(X),2), fn1, fn2)

        Xscaled = X/l
        Xs = tf.reduce_sum(tf.square(Xscaled), 1) # [n] or [n,N]
        Xt = tf.transpose(Xscaled) # [N,d,n] or [d,n]
        if X2 is None:
            # batched case : [N,n,d]@[N,d,n]->[N,n,n]->[n,n,N]
            # non-batch case:[n,d]@[d,n]->[n,n]->[n,n]
            return -2*tf.transpose(tf.batch_matmul(Xt, Xt, adj_x=True)) + \
                        tf.expand_dims(Xs, 1) + tf.expand_dims(Xs, 0)
        else:
            X2scaled = X2/l
            X2s = tf.reduce_sum(tf.square(X2scaled), 1)
            X2t = tf.transpose(X2scaled) # [N,d,n2] or [d,n2]
            # batched case : [N,n2,d]@[N,d,n]->[N,n2,n]->[n,n2,N]
            # non-batch case:[n,d]@[d,n]->[n,n]->[n,n]
            return -2*tf.transpose(tf.batch_matmul(X2t, Xt, adj_x=True)) + \
                        tf.expand_dims(Xs, 1) + tf.expand_dims(X2s, 0)


    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X):
        # callable to absorb the X-rank difference
        # for non-batch X
        def fn1(): return tf.ones([tf.shape(X)[0]], dtype=float_type)
        # for batch X
        def fn2(): return tf.ones([tf.shape(X)[0],tf.shape(X)[-1]], dtype=float_type)
        return tf.cond(tf.equal(tf.rank(X), 2),fn1,fn2)

    def Cholesky(self, X):
        """
        Cholesky decomposition of K(X).
        If X is sized [n,n], this returns [n,n].
        If X is sized [n,n,N], this returns [n,n,N] where each [n,n] matrices
        are lower triangular.
        """
        jitter = settings.numerics.jitter_level
        # callable to absorb the X-rank difference
        # for non-batch X
        def fn1(): return tf.cholesky(self.K(X)+eye(tf.shape(X)[0])*jitter)
        # for batch X.
        # This is wrong for non-batch case, but can be executed without error.
        perm = tf.concat(concat_dim=0, values=[tf.range(1,tf.rank(X)), [0]]) # [1,2,0] or [1,0]
        def fn2(): return tf.transpose(tf.cholesky(
                        tf.transpose(self.K(X)) +\
                        eye(tf.shape(X)[0])* jitter), perm) # [1,2,0] or [0,1]
        return tf.cond(tf.equal(tf.rank(X), 2), fn1, fn2)

class UnitRBF(UnitStationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def K(self, X, X2=None):
        return tf.exp(-self.square_dist(X, X2)/2)
