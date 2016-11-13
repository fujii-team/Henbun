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
import pandas as pd
import tensorflow as tf
from functools import reduce
from . import transforms, priors
from .param import Variable, graph_key, Parameterized
from .scoping import NameScoped
from ._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class Variational(Parameterized):
    """
    The base class for the Variational parameters.
    """
    def __init__(self, shape, n_layers=[], q_shape='diagonal', prior=None,
         transform=transforms.Identity(), collections=[tf.GraphKeys.VARIABLES]):
        """
        shape: shape of this variational parameters
        n_layers: number of layers
        q_shape: one of 'diagonal' or 'fullrank'
                If 'fullrank' is specified, correlation among 'shape' will be
                considered.
        prior: prior of the variational parameters.
        """
        Parameterized.__init__(self)
        self._shape = list([shape]) if isinstance(shape, int) else list(shape)
        self.n_layers = list([n_layers]) if isinstance(n_layers, int) else list(n_layers)
        self.size = int(reduce(np.multiply, self._shape))
        self.collections = collections
        # for the variational parameters
        assert(q_shape in ['diagonal', 'fullrank'])
        self.q_shape = q_shape
        self.q_mu = Variable(self.size, n_layers=n_layers, collections=collections)
        if self.q_shape is 'diagonal':
            # In the diagonal case, log(q_sqrt) will be stored.
            # (manual transform will be adopted)
            self.q_sqrt = Variable(self.size,
                                    n_layers=n_layers, collections=collections)
        else:
            self.q_sqrt = Variable([self.size,self.size],
                                    n_layers=n_layers, collections=collections)
        # transform and prior
        self.transform = transform
        self.prior = prior
        # sampling is made if this is not LOCAL parameters
        if self.collections is not graph_key.LOCAL:
            # samples from i.i.d
            sample_shape = list(self.n_layers) + [self.size]
            self.u = tf.random_normal(sample_shape, dtype=float_type)
            self._sample()

    def feed(self, x):
        """ sampling is made in this method for the LOCAL case """
        Parameterized.feed(self, x)
        # samples from i.i.d
        sample_shape = list(self.n_layers) + [self.size, tf.shape(x)[-1]]
        self.u = tf.random_normal(sample_shape, dtype=float_type)
        self._sample()

    def _sample(self):
        # Build the sampling Ops
        # samples from the posterior
        with self.tf_mode():
            if self.q_shape is 'diagonal':
                self._tensor = self.q_mu + tf.exp(self.q_sqrt) * self.u
            else:
                self._tensor = self.q_mu + tf.squeeze(tf.batch_matmul(
                    tf.matrix_band_part(self.q_sqrt,-1,0), tf.expand_dims(self.u, -1)),
                    [-1])

    def KL(self, collection):
        if collection in self.collections:
            return self._KL()
        else:
            return np.zeros(1, dtype=np_float_type)

    @property
    def tensor(self):
        """
        Returns samples from the posterior
        """
        if self.collections is graph_key.LOCAL:
            shape = self.n_layers + self.shape + [-1]
        else:
            shape = self.n_layers + self.shape
        return tf.reshape(self._tensor, shape)

    @property
    def logdet(self):
        """
        Returns the log-determinant of the posterior
        """
        if self.q_shape is 'diagonal':
            return 2*self.q_sqrt # size [*shape]
        else:
            return tf.log(tf.square(tf.matrix_diag_part(self.q_sqrt)))

    def _KL(self):
        #  E_{q(f)} [log q(f)]
        kl = - 0.5 * tf.reduce_sum(np.log(2.0*np.pi) + self.logdet + tf.square(self.u))
        # - E_{q(f)}[log p(f)]
        if self.prior is not None:
            kl -= self.prior.logp(self._tensor)
            kl -= tf.reduce_sum(self.transform.tf_log_jacobian(self._tensor))
        return kl

""" --- predefined variational parameters ---  """

class Normal(Variational):
    """
    Variational parameters without transform and Normal prior.
    """
    def __init__(self, shape, n_layers=[], q_shape='diagonal',
                                        collections=[tf.GraphKeys.VARIABLES]):
        Variational.__init__(self, shape, q_shape=q_shape, n_layers=n_layers,
                        prior=priors.Normal(), transform=transforms.Identity(),
                        collections=collections)
    def _KL(self):
        """
        Overwrite _KL method to increase efficiency.
        """
        return - 0.5 * tf.reduce_sum(self.logdet + tf.square(self.u) \
                                               - tf.square(self._tensor))
