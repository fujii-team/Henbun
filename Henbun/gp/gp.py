import numpy as np
import tensorflow as tf
from ..param import Variable, Parameterized, graph_key
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
        return tf.batch_matmul(u, L, adj_y=True) # sized [N,n]


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

        # Non-batch case x:[n,d]. Return shape [N,n]
        if x.get_shape().ndims==2:
            # Effective upper-triangular cholesky factor L^T
            LnT = tf.matrix_triangular_solve(Lm, self.kern.K(self.z, x)) # sized [m,n]
            samples = tf.batch_matmul(u, LnT) # sized [N,n]
            # additional variance due to the sparse approximation.
            if q_shape is 'diagonal':
                diag_var = jitter + self.kern.Kdiag(x) - tf.reduce_sum(tf.square(LnT), -2)
                return samples + \
                    tf.sqrt(tf.abs(diag_var)) * tf.random_normal(tf.shape(x)[:-1], dtype=float_type)
            elif q_shape is 'neglected':
                return samples
            else: # fullrank
                Knn = self.kern.K(x) # [n,n]
                jitterI = tf.eye(tf.shape(x)[-2]) * jitter*2
                chol = tf.cholesky(Knn - tf.batch_matmul(LnT, LnT, adj_x=True) + jitterI) # [n,n]
                return samples + tf.matmul(
                            tf.random_normal([N,tf.shape(x)[0]], dtype=float_type), # [N,n]
                            chol, transpose_b=True) # [N,n]@[n,n] -> [N,n]

        # Batch case. x:[N,n,d]. Return shape [N,n]
        elif x.get_shape().ndims==3:
            z = tf.tile(tf.expand_dims(self.z, 0), [N,1,1]) # [N,m,d]
            # Effective upper-triangular cholesky factor L^T
            # Cholesky factor (upper triangluar) of K(z)^-1
            '''
            LminvT = tf.matrix_triangular_solve(Lm, tf.eye(self.m)) # [m,m]
            LnT = tf.batch_matmul(tf.tile(tf.expand_dims(LminvT, 0), [N,1,1]),
                            self.kern.K(z, x), adj_x=True) # [N,m,n]
            '''
            # TODO Do not understand why but the above fails in the following Cholesky factorization.
            # The below is an equivalent but much slower calculation.
            LnT = tf.matrix_triangular_solve(tf.tile(tf.expand_dims(Lm, 0), [N,1,1]),
                            self.kern.K(z, x)) # [N,m,n]

            samples = tf.squeeze(tf.batch_matmul(tf.expand_dims(u,1), LnT), [1]) # [N,1,m]*[N,m,n]->[N,n]
            if q_shape is 'diagonal':
            # additional variance due to the sparse approximation.
                diag_var = jitter + self.kern.Kdiag(x) - tf.reduce_sum(tf.square(LnT), -2) # [N,n]
                return samples + \
                    tf.sqrt(tf.abs(diag_var)) * tf.random_normal(tf.shape(x)[:-1], dtype=float_type)
            elif q_shape is 'neglected':
                return samples
            else: # fullrank case
                Knn = self.kern.K(x) # [N,n,n]
                jitterI = tf.eye(tf.shape(x)[-2]) * jitter*2
                chol = tf.cholesky(Knn - tf.batch_matmul(LnT, LnT, adj_x=True) + jitterI) # [N,n,n]
                return samples + tf.squeeze(tf.batch_matmul(
                    tf.random_normal([N, 1,tf.shape(x)[1]], dtype=float_type),
                    chol, adj_y=True), [1])

        raise ValueError('shape is not specified for tensor x')
