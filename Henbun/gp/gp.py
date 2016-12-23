import numpy as np
import tensorflow as tf
from ..param import Variable, Parameterized, graph_key
from ..tf_wraps import eye
from .._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class GP(Parameterized):
    """
    A vanila implementation, in order to sample from the Gaussian Process
    posterior.

    The posterior mean and covariance are represented by
    + mean: L*u.q_mu
    + covariance: (L*u.q_sqrt)*(L*u.q_sqrt)^T
    where
    L is cholesky factor of the GP kernel, K(x,x) = L L^T
    u.q_mu is the mean of the whitened variational posterior, u
    u.q_sqrt is the cholesky factor the whitened variational posterior, u,
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
        + u: Variational parameters, such as Normal or Gaussian, sized [n, N].
        returns:
        + samples: Samples from the posterior sized [n,N]

        Note:
        The size of the first axis of x and u should be the same.
        """
        L = self.kern.Cholesky(x) # sized [n,n]
        return tf.matmul(L, u) # sized [n,N]


class SparseGP(GP):
    """
    The sparse approximation of the Gaussian Process posterior.
    This class pocesses the position of the inducing variables, z sized [m,d].

    With given whitened variational posterior u, and coordinates x sized [n,d],
    the variational posterior is represented by N(m,S)
    with
    m = Knm*Lm^-T*u.q_mu
    S = Knm*Kmm^-1*Kmn + (Knm*Lm^-T*u.q_sqrt)^2

    where
    Knm = K(x,z)          sized [N,n,m]
    Kmm = K(z,z)          sized [m,m]
    Lm = cholesky(K(z,z)) sized [m,m]

    Due to the sparse approximation, an additional variance should be considered,
    Knm*Kmm^-1*Kmn
    We support the following three approximations,
    + diagonal: The first term of S is approximated by diagonalization.
                This option is default.
    + neglected: The first term of S is neglected.
                Do not use this option in the learning phase, because it removes
                the z dependence of ELBO.
    + fullrank: The first term of S is fully factorized.
                This option might be very slow and GP class may be faster.

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

    def samples(self, x, u, q_shape='diagonal'):
        """
        Returns samples from GP.
        args:
        + x: coordinate variables, sized [n,d] or [n,d,N].
        + u: inducing point values, sized [m,N]
        + q_shape: How to approximate the covariance, Knn-Knm Kmm^-1 Kmn term.
                Shoule be one of ['diagonal', 'neglect', 'fullrank'].
                'diagonal': Diagonalize this term (default).
                'neglected' : Neglect this term. Do not use in the learning phase.
                            (No dependence on z).
                'fullrank': Fully diagonalize this term.
                            Very slow.
        """
        assert(q_shape in ['diagonal','neglected','fullrank'])
        jitter = settings.numerics.jitter_level
        N = tf.shape(u)[-1]
        # Cholesky factor of K(z,z)
        Lm = self.kern.Cholesky(self.z) # sized [m,m]
        I = eye(tf.shape(self.z)[0])
        Lminv = tf.transpose(tf.matrix_triangular_solve(Lm, I)) # [m,m]

        # Non-batch case x:[n,d]. Return shape [n,N]
        def sample():
            Ln = tf.matmul(self.kern.K(x, self.z), Lminv) # sized [n,m]
            samples = tf.matmul(Ln, u) # sized [n,N]
            # additional variance due to the sparse approximation.
            if q_shape is 'diagonal':
                diag_var = jitter + \
                    tf.expand_dims(self.kern.Kdiag(x) - tf.reduce_sum(tf.square(Ln), -1), -1) # [n,1]
                return samples + \
                    tf.sqrt(diag_var) * tf.random_normal([tf.shape(x)[0], N],
                                                dtype=float_type)
            elif q_shape is 'neglected':
                return samples
            else: # fullrank
                Knn = tf.tile(tf.expand_dims(self.kern.K(x),0), [N,1,1])
                jitterI = tf.tile(tf.expand_dims(eye(tf.shape(x)[0]),0), [N,1,1])
                chol = tf.cholesky(Knn - tf.batch_matmul(Ln, Ln, adj_y=True) + jitterI) # [N,n,n]
                return samples + \
                    tf.transpose(
                        tf.squeeze(tf.batch_matmul(
                            chol, tf.random_normal([N, tf.shape(x)[0], 1], dtype=float_type)),
                            [-1])) # [N,n,1] -> [N,n] -> [n,N]

        # Batch case. x:[n,d,N]. Return shape [n,N]
        def sample_batch():
            z = tf.tile(tf.expand_dims(self.z, -1), [1,1,N])
            Ln = tf.batch_matmul(self.kern.K(x, z), tf.tile(tf.expand_dims(Lminv, 0), [N,1,1]))
            samples = tf.transpose(tf.squeeze( # [N,n,1] -> [n,N]
                    tf.batch_matmul(Ln, tf.expand_dims(tf.transpose(u), -1)), [-1]))
            if q_shape is 'diagonal':
            # additional variance due to the sparse approximation.
                diag_var = jitter + \
                    tf.transpose(self.kern.Kdiag(x) - tf.reduce_sum(tf.square(Ln), -1))
                return samples + \
                    tf.sqrt(diag_var) * tf.random_normal([tf.shape(x)[0], N], dtype=float_type)
            elif q_shape is 'neglected':
                return samples
            else: # fullrank case
                Knn = self.kern.K(x)
                jitterI = tf.tile(tf.expand_dims(eye(tf.shape(x)[0]),0), [N,1,1])
                chol = tf.cholesky(Knn - tf.batch_matmul(Ln, Ln, adj_y=True) + jitterI) # [N,n,n]
                return samples + \
                    tf.transpose(
                        tf.squeeze(tf.batch_matmul(
                            chol, tf.random_normal([N, tf.shape(x)[0], 1], dtype=float_type)),
                            [-1])) # [N,n,1] -> [N,n] -> [n,N]

        return tf.cond(tf.equal(tf.rank(x),2), sample, sample_batch)
