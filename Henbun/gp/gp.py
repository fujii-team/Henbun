import numpy as np
import tensorflow as tf
from ..param import Variable, Parameterized, graph_key
from ..tf_wraps import eye
from .._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class GP(Parameterized):
    """
    An implementation to sample from the Gaussian Process posterior.

    The posterior mean and covariance are represented by
    + mean: L*u.q_mu
    + covariance: (L*u.q_sqrt)*(L*u.q_sqrt)^T
    where
    L is cholesky factor of the GP kernel, K(x,x) = L L^T
    u.q_mu is the mean of the variational posterior, u
    u.q_sqrt is the cholesky factor of the variational posterior, u,
    u ~ N(q_mu, q_sqrt*q_sqrt^T)

    This class does not consider mean functions.
    mean_functions should be added manually.

    The typical usage is
    >>> gp = GP(hb.kernels.UnitRBF())
    >>> u = hb.variationals.Normal(shape=[n,N])
    >>> samples = gp.samples(x, u)

    Note:
    Use SparseGP below to get large amount of samples.
    """
    def __init__(self, kern):
        Parameterized.__init__(self)
        self.kern = kern

    def samples(self, x, u):
        """
        Draw samples from the posterior, with given coordinate variables x.
        args:
        + x: Coordinate variables sized [n, d].
        + u: Variational parameters, such as Normal or Gaussian, sized [N,n].
        returns:
        + samples: Samples from the posterior sized [n,N]

        Note:
        The size of the first axis of x and u should be the same.
        """
        L = self.kern.Cholesky(x) # sized [n,n]
        return tf.matmul(u, L, transpose_b=True) # sized [N,n]


class SparseGP(GP):
    """
    The sparse approximation of the Gaussian Process posterior.
    This class pocesses the position of the inducing variables, z sized [m,d].

    With given whitened variational posterior u, and coordinates x sized [n,d],
    the variational posterior is represented by N(m,S)
    with
    m = Knm*Lm^-T*u.q_mu
    S = (Knn-Knm*Kmm^-1*Kmn) + (Knm*Lm^-T*u.q_sqrt)^2

    where
    Knm = K(x,z)          sized [N,n,m]
    Kmm = K(z,z)          sized [m,m]
    Lm = cholesky(K(z,z)) sized [m,m]

    Due to the sparse approximation, an additional variance should be considered,
    Knn-Knm*Kmm^-1*Kmn
    We support the following three approximations,
    + diagonal: The first term of S is approximated by diagonalization.
                This option is default.
    + neglected: The first term of S is neglected.
    + fullrank: The first term of S is fully factorized.
                This option might be very slow and even GP class may be faster.

    This class does not consider mean functions.
    mean_functions should be added manually.

    The typical usage is
    >>> gp = SparseGP(kern=hb.kernels.UnitRBF(), z=z)
    >>> u = hb.variationals.Normal(shape=[n,N])
    >>> samples = gp.samples(x, u)
    """
    def __init__(self, kern, z, collections=[graph_key.VARIABLES]):
        """
        args
        + kern: kernel object
        + z: 2d-np.array indicating the initial values for inducing point locations.
        + collections: variable collections for z
        """
        GP.__init__(self, kern)
        # inducing points
        self.z = Variable(shape=z.shape, collections=collections)
        self.z = z # set the inital value
        self.m = len(z)

    def samples(self, x, u, q_shape='diagonal'):
        """
        Returns samples from GP.
        args:
        + x: coordinate variables, sized [n,d] or [N,n,d].
        + u: inducing point values, sized [N,m]
        + q_shape: How to approximate the covariance, Knn-Knm Kmm^-1 Kmn term.
                Shoule be one of ['diagonal', 'neglect', 'fullrank'].
                'diagonal': Diagonalize this term (default).
                'neglected' : Neglect this term.
                'fullrank': Fully factorize this term.
                            Very slow.
        """
        assert(q_shape in ['diagonal','neglected','fullrank'])
        jitter = settings.numerics.jitter_level
        N = tf.shape(u)[0]
        # Cholesky factor of K(z,z)
        Lm = self.kern.Cholesky(self.z) # sized [m,m]

        # TODO insert assertion for shape difference

        LnT = self._effective_LT(x)
        if x.get_shape().ndims==2:
            samples = tf.matmul(u, LnT) # sized [N,n]
        elif x.get_shape().ndims==3:
            samples = tf.squeeze(
                tf.matmul(tf.expand_dims(u,1), LnT), [1]) # [N,1,m]*[N,m,n]->[N,n]

        if q_shape is 'neglected':
            return samples
        elif q_shape is 'diagonal':
            diag_cov = self._additional_cov(x, LnT, 'diagonal')
            return samples + tf.sqrt(tf.abs(diag_cov)) \
                        * tf.random_normal(tf.shape(x)[:-1], dtype=float_type)
        else: # 'fullrank'
            jitterI = eye(tf.shape(x)[-2]) * jitter*2
            chol = tf.cholesky(self._additional_cov(x, LnT, 'fullrank') + jitterI) # [n,n]
            if x.get_shape().ndims==2:
                return samples + tf.matmul(
                    tf.random_normal([N,tf.shape(x)[0]], dtype=float_type), # [N,n]
                    chol, transpose_b=True) # [N,n]@[n,n] -> [N,n]
            elif x.get_shape().ndims==3:
                return samples + tf.squeeze(tf.matmul(
                    tf.random_normal([N, 1,tf.shape(x)[1]], dtype=float_type),
                    chol, transpose_b=True), [1])


    def _effective_LT(self, x):
        """
        Returns the effective cholesky factor,
              - T       - 1
        K   L      =  L     K
         nm                  mn
                 T
        with L L   = K
                      mm
        args:
        + x : coordinate for the prediction, sized [N,n,d] or [n,d]
        """
        # Cholesky factor of K(z,z)
        Lm = self.kern.Cholesky(self.z) # sized [m,m]
        if x.get_shape().ndims==2:
            # [m,n] -> [n,m]
            return tf.matrix_triangular_solve(Lm, self.kern.K(self.z, x))
            #Lminv = tf.matrix_triangular_solve(Lm, eye(self.m)) # [m,m]
            #return tf.matmul(Lminv, self.kern.K(self.z, x)) # [m,m]@[m,n]->[m,n]

        # batch case
        elif x.get_shape().ndims==3:
            N = tf.shape(x)[0]
            Lminv = tf.matrix_triangular_solve(Lm, eye(self.m)) # [m,m]
            z = tf.tile(tf.expand_dims(self.z, 0), [N,1,1])
            return tf.matmul(tf.tile(tf.expand_dims(Lminv, 0), [N,1,1]), #[N,m,m]
                             self.kern.K(z, x)) # [N,m,m]@[N,m,n] -> [N,m,n]

        raise ValueError('shape is not specified for tensor x')


    def _additional_cov(self, x, LnT, q_shape):
        """
        Returns the additional GP covariance due to the sparse approximation.
        Knn-Knm*Kmm^-1*Kmn

        args:
        + x : coordinate for the prediction, sized [N,n,d] or [n,d]
        + LnT: Effective Cholesky factor, L^-1 Kmn
        + q_shape: 'diagonal' or 'fullrank'. If it is 'diagonal', it returns
        only the diagonal part of the covariance.
        """
        if q_shape is 'diagonal':
            return self.kern.Kdiag(x) - tf.reduce_sum(tf.square(LnT), -2) # [N,n]
        else:
            Knn = self.kern.K(x) # [N,n,n] or [n,n]
            return Knn - tf.matmul(LnT, LnT, transpose_a=True)
