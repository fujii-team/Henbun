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
from functools import reduce
from .tf_wraps import clip
from . import transforms, priors, densities
from .param import Variable, graph_key, Parameterized
from .scoping import NameScoped
from ._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Variational(Parameterized):
    """
    The base class for the Variational parameters.
    We assume multivariate gaussian distribution for the variational
    distribution.

    The shape of covariance can be chosen from ['diagonal', 'fullrank'].
    In fullrank case, the correlation between 'shape' dimension will be
    considered (This axes are flattened in this class).
    """
    def __init__(self, shape, n_layers=[], n_batch=None,
        q_shape='diagonal', mean=0.0, stddev=1.0,
        prior=None, transform=transforms.Identity(),
        collections=[graph_key.VARIABLES]):
        """
        - shape: list or tuples indicating the shape of this parameters.
                In the LOCAL case, the right most axis is MinibatchSize.
                This axis can be None. In this case, we do not validate the shape.

        - n_layers: List of integers indicating number of layers.

        - n_batches: Integer representing number of batches. It can be None.
                In Local case, the batch_size is automatically determined if
                None is given. If a certain value is specified, then Local and
                Global variables behave same.

        The shape of this variables will be [*n_layers, *n_batches, *shape],
        as param.Variable

        - mean: initial mean of the variational parameters.

        - stddev: initial stddev of the variational parameters

        - prior: prior of the variational parameters.

        - transform: transform of the variational variable.
            Note that the prior is applied to the post-transformation of the variable.

        - q_shape: one of 'diagonal' or 'fullrank'
                If 'fullrank' is specified, correlation among 'shape' will be
                considered.

        - collections: collections for the variational parameters.
        """
        Parameterized.__init__(self)
        self._shape = list([shape]) if isinstance(shape, int) else list(shape)
        self.n_layers = list([n_layers]) if isinstance(n_layers, int) else list(n_layers)
        self.n_batch = n_batch
        self.size = int(reduce(np.multiply, self._shape))
        self.collections = collections
        # for the variational parameters
        assert(q_shape in ['diagonal', 'fullrank'])
        self.q_shape = q_shape
        self.q_mu = Variable(self.size, n_layers=n_layers, n_batch=self.n_batch,
                                mean=mean, stddev=0.1*stddev,
                                collections=collections)
        if self.q_shape is 'diagonal':
            # In the diagonal case, log(q_sqrt) will be stored.
            # (manual transform will be adopted)
            self.q_sqrt = Variable(self.size, n_layers=n_layers, n_batch=self.n_batch,
                                mean=np.log(stddev), stddev=0.1,
                                collections=collections)
        else:
            self.q_sqrt = Variable([self.size,self.size], n_layers=n_layers, n_batch=self.n_batch,
                                mean=stddev, stddev=0.1*stddev,
                                collections=collections)
        # transform and prior
        self.transform = transform
        self.prior = prior
        # sampling is made if this is not LOCAL parameters
        if self.collections is not graph_key.LOCAL:
            if self.n_batch is None:
                sample_shape = list(self.n_layers) + [self.size]
            else:
                sample_shape = list(self.n_layers) + [self.n_batch] + [self.size]
            # sample from i.i.d.
            self.u = tf.random_normal(sample_shape, dtype=float_type)
            with self.tf_mode():
                self._tensor = self._sample(self.u)
                self.transformed_tensor = self.transform.tf_forward(self._tensor)

    def tensor(self):
        """
        In tf_mode, this class is seen as a sample from the variational distribution.
        """
        if self.collections is not graph_key.LOCAL and self.n_batch is None:
            return clip(tf.reshape(self.transformed_tensor, self.n_layers + self._shape))
        else:
            return clip(tf.reshape(self.transformed_tensor, self.n_layers + [-1] + self._shape))

    def feed(self, x):
        """ sampling is made in this method for the LOCAL case """
        Parameterized.feed(self, x)
        if self.collections is graph_key.LOCAL:
            # samples from i.i.d
            sample_shape = self.n_layers + [tf.shape(x)[-2], self.size]
            self.u = tf.random_normal(sample_shape, dtype=float_type)
            self._tensor = self._sample(self.u)
            self.transformed_tensor = self.transform.tf_forward(self._tensor)

    def _sample(self, u):
        """
        Method to sample from the variational distribution.
        """
        # Build the sampling Ops
        # samples from the posterior
        # u: i.i.d. sample
        if self.q_shape is 'diagonal':
            if self._tf_mode:
                return self.q_mu + tf.exp(self.q_sqrt) * u
            else:
                return self.q_mu + tf.exp(self.q_sqrt._tensor) * u
        else:
            if self._tf_mode:
                sqrt = tf.matrix_band_part(self.q_sqrt,-1,0)
                return self.q_mu + tf.einsum(self._einsum_matmul(), sqrt, u)
            else:
                sqrt = tf.matrix_band_part(self.q_sqrt._tensor,-1,0)
                return self.q_mu._tensor + tf.einsum(self._einsum_matmul(), sqrt, u)

    def _einsum_matmul(self):
        """
        A simple method to generate einsum index.
        This method is called in _sample() method with 'fullrank' variational
        parameters.
        """
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        n = len(self.n_layers)
        if self.collections is not graph_key.LOCAL and self.n_batch is None:
            # layer:'..i', shape:'jk'
            # '...ijk,...ik->...ij'
            index1 = alphabet[:n+2] # '...ijk'
            index2 = alphabet[:n]+alphabet[n+1] # '...ik'
            index3 = alphabet[:n+1] # '...ij'
            return index1+','+index2+'->'+index3
        else:
            # layer:'..i', batch:'j', shape:'kl'
            # '...ijkl,...ijl->...,ijk'
            index1 = alphabet[:n+3] # '...ijkl'
            index2 = alphabet[:n+1]+alphabet[n+2] # '...ijl'
            index3 = alphabet[:n+2] # '...ijk'
            return index1+','+index2+'->'+index3

    @property
    def logdet(self):
        """
        Returns the log-determinant of the posterior
        """
        if self.q_shape is 'diagonal':
            return 2.0*self.q_sqrt # size [*shape]
        else:
            return tf.log(tf.square(tf.matrix_diag_part(self.q_sqrt)))

    def KL(self, collection=None):
        """
        Returns the KL values of this variational parameter.
        If collection is None, KL is returned regardless of self.collections.
        """
        if collection is None or collection in self.collections:
            return self._KL()
        else:
            return np.zeros([], dtype=np_float_type)

    def _KL(self):
        """
        Returns the sum of Kulback-Leibler divergence for this variational object.
        This value is gathered by Parameterized.KL()
        """
        #  E_{q(f)} [log q(f)]
        kl = - 0.5 * tf.reduce_sum(np.log(2.0*np.pi) + self.logdet + tf.square(self.u))
        # - E_{q(f)}[log p(f)]
        if self.prior is not None:
            kl -= tf.reduce_sum(self.prior.logp(self.transformed_tensor))
            kl -= tf.reduce_sum(self.transform.tf_log_jacobian(self._tensor))
        return kl

""" --- predefined variational parameters ---  """

class Normal(Variational):
    """
    Variational parameters with Normal prior without transformation.
    """
    def __init__(self, shape, n_layers=[], n_batch=None, q_shape='diagonal',
                                        mean=0.0, stddev=1.0,
                                        collections=[graph_key.VARIABLES]):
        Variational.__init__(self, shape, q_shape=q_shape, n_layers=n_layers,
                        n_batch=n_batch,
                        mean=mean, stddev=stddev,
                        prior=priors.Normal(), transform=transforms.Identity(),
                        collections=collections)
    def _KL(self):
        """
        Overwrite _KL method to increase efficiency.
        """
        return - 0.5 * tf.reduce_sum(self.logdet + tf.square(self.u) \
                                               - tf.square(self._tensor))

class Gaussian(Normal):
    """
    Variational parameters with Gaussian prior without transformation.
    """
    def __init__(self, shape, n_layers=[], n_batch=None, q_shape='diagonal',
                mean=0.0, stddev=1.0, collections=[graph_key.VARIABLES],
                scale_shape=None, scale_n_layers=None):
        """
        - shape: list or tuples indicating the shape of this parameters.
                In the LOCAL case, the right most axis is MinibatchSize.
                This axis can be None. In this case, we do not validate the shape.

        - n_layers: List of integers indicating number of layers.

        - n_batches: Integer representing number of batches. It can be None.
                In Local case, the batch_size is automatically determined if
                None is given. If a certain value is specified, then Local and
                Global variables behave same.

        - q_shape: one of 'diagonal' or 'fullrank'
                If 'fullrank' is specified, correlation among 'shape' will be
                considered.

        - collections: collections for the variational parameters.

        - scale_shape, scale_n_layers: list (or tuple) of integers indicating
                        the shape of scale parameters.
                        By default, the scale shape is
                        [1, 1, 1, 1, 1]
                        with self.n_layers = [l1,l2]
                             self.shape    = [s1,s2,s3]
        """
        # initialize mean and stddev.
        # This class mainly controls scale rather than the variational part.
        if np.abs(mean) < stddev:
            scale_mean = stddev
            q_mean= mean/stddev
            q_std = 1.0
        else:
            scale_mean = np.abs(mean)
            q_mean= 1.0
            q_std = stddev/np.abs(mean)
        #
        Variational.__init__(self, shape, q_shape=q_shape, n_layers=n_layers,
                        n_batch=n_batch,
                        mean=q_mean, stddev=q_std,
                        prior=priors.Normal(), transform=transforms.Identity(),
                        collections=collections)

        # scale shape
        scale_shape = scale_shape or [1 for s in self._shape]
        # scale layer
        scale_layer = scale_n_layers or [1 for s in self.n_layers]
        # Define scale parameter
        self.scale = Variable(scale_shape, n_layers = scale_layer, n_batch=n_batch,
            mean=scale_mean, stddev=0.1*scale_mean, transform=transforms.positive,
                                 collections=collections)

    def tensor(self):
        return self.scale * Normal.tensor(self)

class OffsetGaussian(Gaussian):
    """
    Variational parameters with Gaussian prior with offset.
    """
    def __init__(self, shape, n_layers=[], n_batch=None, q_shape='diagonal',
                mean=0.0, stddev=1.0, collections=[graph_key.VARIABLES],
                scale_shape=None, scale_n_layers=None):

        Gaussian.__init__(self,
            shape=shape, n_layers=n_layers, n_batch=n_batch,
            q_shape=q_shape, mean=0.0, stddev=stddev, collections=collections,
            scale_shape=scale_shape, scale_n_layers=scale_n_layers)
        # scale shape
        offset_shape = scale_shape or [1 for s in self._shape]
        # scale layer
        offset_layer = scale_n_layers or [1 for s in self.n_layers]
        # Define scale parameter
        self.offset = Variable(offset_shape, n_layers = offset_layer, n_batch=n_batch,
            mean=mean, stddev=0.1*mean, collections=collections)

    def tensor(self):
        return Gaussian.tensor(self) + self.offset

class Beta(Variational):
    """
    Variational parameters with Beta prior.
    The variational tensor is mapped to (0,1) space by Logistic function.
    The beta prior is assumed for this distribution, where its hyperparameter
    alpha and beta are also Variables.
    """
    def __init__(self, shape, n_layers=[], n_batch=None, q_shape='diagonal',
                mean=0.0, stddev=1.0, collections=[graph_key.VARIABLES],
                scale_shape=None, scale_n_layers=None):
        """
        - shape: list or tuples indicating the shape of this parameters.
                In the LOCAL case, the right most axis is MinibatchSize.
                This axis can be None. In this case, we do not validate the shape.

        - n_layers: List of integers indicating number of layers.

        - n_batches: Integer representing number of batches. It can be None.
                In Local case, the batch_size is automatically determined if
                None is given. If a certain value is specified, then Local and
                Global variables behave same.

        - q_shape: one of 'diagonal' or 'fullrank'
                If 'fullrank' is specified, correlation among 'shape' will be
                considered.

        - collections: collections for the variational parameters.

        - scale_shape, scale_n_layers: list (or tuple) of integers indicating
                        the shape of hyper parameters.
                        By default, the scale shape is
                        [1, 1, 1, 1, 1]
                        with self.n_layers = [l1,l2]
                             self.shape    = [s1,s2,s3]
        """
        # map mean and stddev
        Variational.__init__(self, shape, q_shape=q_shape, n_layers=n_layers,
                        n_batch=n_batch,
                        mean=mean, stddev=stddev,
                        transform=transforms.Logistic(),
                        collections=collections)
        # scale shape
        scale_shape = scale_shape or [1 for s in self._shape]
        # scale layer
        scale_layer = scale_n_layers or [1 for s in self.n_layers]
        # Define scale parameter
        self.alpha = Variable(scale_shape, n_layers = scale_layer, n_batch=n_batch,
                            mean=1.0, stddev=0.1, transform=transforms.positive,
                            collections=collections)
        self.beta = Variable(scale_shape, n_layers = scale_layer, n_batch=n_batch,
                            mean=1.0, stddev=0.1, transform=transforms.positive,
                            collections=collections)

    def _KL(self):
        """
        Returns the sum of Kulback-Leibler divergence for this variational object.
        This value is gathered by Parameterized.KL()
        """
        #  E_{q(f)} [log q(f)]
        kl = - 0.5 * tf.reduce_sum(np.log(2.0*np.pi) + self.logdet + tf.square(self.u))
        # - E_{q(f)}[log p(f)]
        if self.prior is not None:
            kl -= tf.reduce_sum(densities.beta(self.alpha, self.beta,
                                    self.transformed_tensor))
            kl -= tf.reduce_sum(self.transform.tf_log_jacobian(self._tensor))
        return kl
