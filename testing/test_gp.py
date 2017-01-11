from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32

class test_gp_numerics(unittest.TestCase):
    def test(self):
        """ Make sure SpareseGP works with very small jitter"""
        self.rng = np.random.RandomState(0)
        self.m = hb.model.Model()
        # sparse gp
        self.m.sparse_gp = hb.gp.SparseGP(z= np.random.randn(600,1),
            kern=hb.gp.kernels.UnitRBF(lengthscales=np.ones(1, np_float_type)))
        # variational posterior
        self.m.u = hb.variationals.Normal(shape=[1,600])
        # new coordinate
        x = tf.constant(self.rng.randn(400,1), dtype=float_type)
        #self.m.initialize()
        with self.m.tf_mode():
            # just test works fine
            samples = self.m.run(self.m.sparse_gp.samples(x, self.m.u, 'neglected'))
            self.assertFalse(np.any(np.isnan(samples))) # test if it is not None
            # Also test for q_shape=diagonal case
            samples = self.m.run(self.m.sparse_gp.samples(x, self.m.u, 'diagonal'))
            self.assertFalse(np.any(np.isnan(samples))) # test if it is not None


class test_gp_dense(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.m = hb.model.Model()
        # dense gp N=20,m=30
        self.m.gp = hb.gp.GP(
            kern=hb.gp.kernels.UnitRBF(lengthscales=np.ones(1, np_float_type)))
        # variational posterior
        self.m.u = hb.variationals.Normal(shape=[20,30])

    def test_dense(self):
        x = tf.constant(self.rng.randn(30,2), dtype=float_type)
        #self.m.initialize()
        with self.m.tf_mode():
            # just test works fine
            samples = self.m.gp.samples(x, self.m.u)
            # make sure gradients certainly works
            grad = tf.gradients(tf.reduce_sum(samples*samples),
                                    self.m.get_tf_variables())
        # assert shape
        self.assertTrue(np.allclose(self.m.run(samples).shape, [20,30]))
        # assert grad certainly works
        gvalues = [self.m.run(g) for g in grad if g is not None]
        self.assertTrue(len(gvalues)>0)


class test_gp_sparse(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.m = hb.model.Model()
        # sparse gp N=20,m=30, d=2
        self.m.sparse_gp = hb.gp.SparseGP(z= np.linspace(-2.0,2.0,60).reshape(-1,2),
            kern=hb.gp.kernels.UnitRBF(lengthscales=np.ones(1, np_float_type)*0.5))
        # variational posterior
        self.m.u = hb.variationals.Normal(shape=[20,30])

    def test_effective_LT(self):
        # If x == z, the effective LT should be identical to Cholesky factorization.
        # --- non-batched case ---
        x = tf.constant(np.linspace(-2.0,2.0,60).reshape(-1,2), dtype=float_type)
        with self.m.tf_mode():
            LT_eff = self.m.run(self.m.sparse_gp._effective_LT(x))
            L = self.m.run(tf.transpose(self.m.sparse_gp.kern.Cholesky(x)))
        self.assertTrue(np.allclose(LT_eff, L, atol=0.005))

        # --- check resutls by non-batch and batched cases are identical
        x = tf.constant(np.linspace(-2.0,2.0,60).reshape(1,-1,2), dtype=float_type)
        with self.m.tf_mode():
            LT1_eff = self.m.run(self.m.sparse_gp._effective_LT(x))
            L1 = self.m.run(tf.transpose(self.m.sparse_gp.kern.Cholesky(x), [0,2,1]))
        self.assertTrue(np.allclose(LT1_eff, LT_eff, atol=0.005))
        self.assertTrue(np.allclose(L1, L, atol=0.005))

        # --- batched case ---
        x = np.array([np.linspace(-2.0,2.0,60).reshape(-1,2) for _ in range(20)])
        x = tf.constant(x, dtype=float_type)
        with self.m.tf_mode():
            LT_eff = self.m.run(self.m.sparse_gp._effective_LT(x))
            L = self.m.run(tf.transpose(self.m.sparse_gp.kern.Cholesky(x), [0,2,1]))
        self.assertTrue(np.allclose(LT_eff, L, atol=0.005))



    def test_additional_cov1(self):
        # If x == z, the additional_cov should be zero.
        # --- non-batched case ---
        x = tf.constant(np.linspace(-2.0,2.0,60).reshape(-1,2), dtype=float_type)
        for q_shape in ['diagonal', 'fullrank']:
            with self.m.tf_mode():
                LT_eff = self.m.sparse_gp._effective_LT(x)
                cov = self.m.run(self.m.sparse_gp._additional_cov(x, LT_eff, q_shape))
            self.assertTrue(np.allclose(cov, 0.0, atol=0.005))

        # --- batched case ---
        x = np.array([np.linspace(-2.0,2.0,60).reshape(-1,2) for _ in range(20)])
        x = tf.constant(x, dtype=float_type)
        for q_shape in ['diagonal', 'fullrank']:
            with self.m.tf_mode():
                LT_eff = self.m.sparse_gp._effective_LT(x)
                cov = self.m.run(self.m.sparse_gp._additional_cov(x, LT_eff, q_shape))
            self.assertTrue(np.allclose(cov, 0.0, atol=0.005))
        pass

    def test_additional_cov1(self):
        # Test the diagonal part of the additional_cov is correct
        # --- non-batched case ---
        x = tf.constant(self.rng.randn(20,2), dtype=float_type)
        with self.m.tf_mode():
            LT_eff = self.m.sparse_gp._effective_LT(x)
            cov = self.m.run(self.m.sparse_gp._additional_cov(x, LT_eff, 'fullrank'))
            cov_diag = self.m.run(self.m.sparse_gp._additional_cov(x, LT_eff, 'diagonal'))
        self.assertTrue(np.allclose(np.diagonal(cov), cov_diag, atol=0.0001))
        # --- batched case ---
        x = tf.constant(self.rng.randn(21,20,2), dtype=float_type)
        with self.m.tf_mode():
            LT_eff = self.m.sparse_gp._effective_LT(x)
            cov = self.m.run(self.m.sparse_gp._additional_cov(x, LT_eff, 'fullrank'))
            cov_diag = self.m.run(self.m.sparse_gp._additional_cov(x, LT_eff, 'diagonal'))
        for i in range(len(cov)):
            self.assertTrue(np.allclose(np.diagonal(cov[i]), cov_diag[i], atol=0.0001))

    def test_non_batch_sparse(self):
        x = tf.constant(self.rng.randn(40,2), dtype=float_type)
        #self.m.initialize()
        with self.m.tf_mode():
            # just test works fine
            samples = self.m.sparse_gp.samples(x, self.m.u)
            # make sure gradients certainly works
            grad = tf.gradients(tf.reduce_sum(samples*samples),
                                    self.m.get_tf_variables())
        # assert shape
        self.assertTrue(np.allclose(self.m.run(samples).shape, [20,40]))
        # assert grad certainly works
        gvalues = [self.m.run(g) for g in grad if g is not None]
        self.assertTrue(len(gvalues)>0)

        # test other approximation methods
        for q_shape in ['neglected', 'fullrank']:
            with self.m.tf_mode():
                # just test works fine
                samples = self.m.sparse_gp.samples(x, self.m.u, q_shape=q_shape)
            # assert shape
            self.assertTrue(np.allclose(self.m.run(samples).shape, [20,40]))

    def test_batch(self):
        # n=200, N=20, d=2
        x = tf.constant(self.rng.randn(20,40,2), dtype=float_type)
        #self.m.initialize()
        with self.m.tf_mode():
            # just test works fine
            samples = self.m.sparse_gp.samples(x, self.m.u)
            # make sure gradients certainly works
            grad = tf.gradients(samples, tf.trainable_variables())
        # assert shape
        self.assertTrue(np.allclose(self.m.run(samples).shape, [20,40]))
        # assert grad certainly works
        gvalues = [self.m.run(g) for g in grad if g is not None]
        self.assertTrue(len(gvalues)>0)
        # test other approximation methods
        for q_shape in ['neglected', 'fullrank']:
            with self.m.tf_mode():
                # just test works fine
                samples = self.m.sparse_gp.samples(x, self.m.u, q_shape=q_shape)
            # assert shape
            self.assertTrue(np.allclose(self.m.run(samples).shape, [20,40]))


class gp(hb.model.Model):
    def setUp(self, n_samples=None):
        rng = np.random.RandomState(0)
        # --- data ---
        self.X = np.linspace(0,6,20).reshape(-1,1)
        self.Y = np.cos(self.X) + rng.randn(20,1)*0.1
        # --- kernel ---
        self.kern = hb.gp.kernels.UnitRBF() # kernel with unit variance
        self.k_var = hb.param.Variable(1, transform=hb.transforms.positive) # kernel variance
        # --- variational parameter ---
        self.q = hb.variationals.Normal(shape=[20], n_samples=n_samples, q_shape='fullrank')
        # --- likelihood variance ---
        self.var = hb.param.Variable(1, transform=hb.transforms.positive)

    @hb.model.AutoOptimize()
    def likelihood_1sample(self):
        """
        Likelihood by variational method with a single sample
        """
        q = tf.expand_dims(self.q, [-2])
        Lq = tf.transpose(tf.matmul(q, self.kern.Cholesky(self.X), transpose_b=True) * tf.sqrt(self.k_var))
        #Lq = tf.matmul(self.kern.Cholesky(self.X), self.q, transpose_b=True) * tf.sqrt(self.k_var)
        return tf.reduce_sum(hb.densities.gaussian(self.Y, Lq, self.var))\
                - self.KL()

    @hb.model.AutoOptimize()
    def likelihood_n_samples(self):
        """
        Likelihood by variational method with n samples option
        """
        Lq = tf.transpose(tf.matmul(self.q, self.kern.Cholesky(self.X), transpose_b=True) * tf.sqrt(self.k_var))
        #Lq = tf.matmul(self.kern.Cholesky(self.X), self.q, transpose_b=True) * tf.sqrt(self.k_var)
        return tf.reduce_sum(hb.densities.gaussian(self.Y, Lq, self.var))\
                - self.KL()

    @hb.model.AutoOptimize()
    def likelihood_ana(self):
        """
        Analytical likelihood
        """
        K = self.k_var * self.kern.K(self.X) + hb.tf_wraps.eye(self.X.shape[0])*self.var
        L = tf.cholesky(K)
        return hb.densities.multivariate_normal(self.Y, tf.zeros_like(self.Y, float_type), L)


class test_gpr(unittest.TestCase):
    def test_1sample(self):
        """
        Variational parameter with a single sample
        """
        tf.set_random_seed(0)
        m = gp()
        # run normal gpr
        m.likelihood_ana().compile(optimizer=tf.train.AdamOptimizer(0.01))
        m.likelihood_ana().optimize(maxiter=2000)
        lik = m.likelihood_ana().run()
        k_lengthscale = m.kern.lengthscales.value
        k_var = m.k_var.value
        var = m.var.value

        # run variational gpr
        # reset hyperparameters
        m.kern.lengthscales=1.0
        m.k_var=1.0
        m.var  =1.0
        # initialize
        m.q.q_sqrt = m.q.q_sqrt.value*0.01
        # adopt an exponential_decay of learning rate to maintain a good convergence.
        m.likelihood_1sample().compile(optimizer=tf.train.AdamOptimizer(0.001))
        m.likelihood_1sample().optimize(maxiter=40000)

        # average samples for likelihood
        lik_var = np.mean([m.likelihood_1sample().run() for i in range(100)])
        print(lik, lik_var)
        print(k_lengthscale, m.kern.lengthscales.value)
        print(k_var, m.k_var.value)
        print(var, m.var.value)
        self.assertTrue(np.allclose(lik, lik_var, atol=1.0))
        self.assertTrue(np.allclose(k_lengthscale, m.kern.lengthscales.value, rtol=0.3))
        self.assertTrue(np.allclose(var, m.var.value, rtol=0.3))

    def test_n_samples(self):
        tf.set_random_seed(0)
        n_samples = 100
        m = gp(n_samples=n_samples)
        # run normal gpr
        m.likelihood_ana().compile(optimizer=tf.train.AdamOptimizer(0.01))
        m.likelihood_ana().optimize(maxiter=2000)
        lik = m.likelihood_ana().run()
        k_lengthscale = m.kern.lengthscales.value
        k_var = m.k_var.value
        var = m.var.value

        # run variational gpr
        # reset hyperparameters
        m.kern.lengthscales=1.0
        m.k_var=1.0
        m.var  =1.0
        # initialize
        m.q.q_sqrt = m.q.q_sqrt.value*0.01
        # adopt an exponential_decay of learning rate to maintain a good convergence.
        m.likelihood_n_samples().compile(optimizer=tf.train.AdamOptimizer(0.001))
        m.likelihood_n_samples().optimize(maxiter=10000)

        # average samples for likelihood
        lik_var = m.likelihood_n_samples().run() / n_samples
        print(lik, lik_var)
        print(k_lengthscale, m.kern.lengthscales.value)
        print(k_var, m.k_var.value)
        print(var, m.var.value)
        self.assertTrue(np.allclose(lik, lik_var, atol=1.0))
        self.assertTrue(np.allclose(k_lengthscale, m.kern.lengthscales.value, rtol=0.3))
        self.assertTrue(np.allclose(var, m.var.value, rtol=0.3))


if __name__ == '__main__':
    unittest.main()
