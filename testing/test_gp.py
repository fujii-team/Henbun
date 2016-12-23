from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32

class test_gp(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.m = hb.model.Model()
        # dense gp
        self.m.gp = hb.gp.GP(
            kern=hb.gp.kernels.UnitRBF(lengthscales=np.ones(1, np_float_type)))
        # sparse gp
        self.m.sparse_gp = hb.gp.SparseGP(z= np.linspace(0,1.0,60).reshape(-1,2),
            kern=hb.gp.kernels.UnitRBF(lengthscales=np.ones(1, np_float_type)))
        # variational posterior
        self.m.u = hb.variationals.Normal(shape=[30,20])

    def test_dense(self):
        x = tf.constant(self.rng.randn(30,2), dtype=float_type)
        self.m.initialize()
        with self.m.tf_mode():
            # just test works fine
            samples = self.m.gp.samples(x, self.m.u)
            # make sure gradients certainly works
            grad = tf.gradients(tf.reduce_sum(samples*samples),
                                    self.m.get_tf_variables())
        # assert shape
        self.assertTrue(np.allclose(self.m.run(samples).shape, [30,20]))
        # assert grad certainly works
        gvalues = [self.m.run(g) for g in grad if g is not None]
        self.assertTrue(len(gvalues)>0)

    def test_non_batch_sparse(self):
        x = tf.constant(self.rng.randn(40,2), dtype=float_type)
        self.m.initialize()
        with self.m.tf_mode():
            # just test works fine
            samples = self.m.sparse_gp.samples(x, self.m.u)
            # make sure gradients certainly works
            grad = tf.gradients(tf.reduce_sum(samples*samples),
                                    self.m.get_tf_variables())
        # assert shape
        self.assertTrue(np.allclose(self.m.run(samples).shape, [40,20]))
        # assert grad certainly works
        gvalues = [self.m.run(g) for g in grad if g is not None]
        self.assertTrue(len(gvalues)>0)

        # test other approximation methods
        for q_shape in ['neglected', 'fullrank']:
            with self.m.tf_mode():
                # just test works fine
                samples = self.m.sparse_gp.samples(x, self.m.u, q_shape=q_shape)
            # assert shape
            self.assertTrue(np.allclose(self.m.run(samples).shape, [40,20]))

    def test_batch(self):
        x = tf.constant(self.rng.randn(40,2,20), dtype=float_type)
        self.m.initialize()
        with self.m.tf_mode():
            # just test works fine
            samples = self.m.sparse_gp.samples(x, self.m.u)
            # make sure gradients certainly works
            grad = tf.gradients(samples, tf.trainable_variables())
        # assert shape
        self.assertTrue(np.allclose(self.m.run(samples).shape, [40,20]))
        # assert grad certainly works
        gvalues = [self.m.run(g) for g in grad if g is not None]
        self.assertTrue(len(gvalues)>0)
        # test other approximation methods
        for q_shape in ['neglected', 'fullrank']:
            with self.m.tf_mode():
                # just test works fine
                samples = self.m.sparse_gp.samples(x, self.m.u, q_shape=q_shape)
            # assert shape
            self.assertTrue(np.allclose(self.m.run(samples).shape, [40,20]))


class gp(hb.model.Model):
    def setUp(self):
        rng = np.random.RandomState(0)
        # --- data ---
        self.X = np.linspace(0,6,20).reshape(-1,1)
        self.Y = np.cos(self.X) + rng.randn(20,1)*0.1
        # --- kernel ---
        self.kern = hb.gp.kernels.UnitRBF() # kernel with unit variance
        self.k_var = hb.param.Variable(1, transform=hb.transforms.positive) # kernel variance
        # --- variational parameter ---
        self.q = hb.variationals.Normal(shape=[20,1], q_shape='fullrank')
        # --- likelihood variance ---
        self.var = hb.param.Variable(1, transform=hb.transforms.positive)

    @hb.model.AutoOptimize()
    def likelihood_var(self):
        """
        Likelihood by variational method
        """
        Lq = tf.matmul(self.kern.Cholesky(self.X), self.q) * tf.sqrt(self.k_var)
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
    def test(self):
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
        # adopt an exponential_decay of learning rate to maintain a good convergence.
        m.likelihood_var().compile(optimizer=tf.train.AdamOptimizer(0.001))
        m.likelihood_var().optimize(maxiter=40000)

        # average samples for likelihood
        lik_var = np.mean([m.likelihood_var().run() for i in range(100)])
        print(lik, lik_var)
        print(k_lengthscale, m.kern.lengthscales.value)
        print(k_var, m.k_var.value)
        print(var, m.var.value)
        self.assertTrue(np.allclose(lik, lik_var, atol=1.0))
        self.assertTrue(np.allclose(k_lengthscale, m.kern.lengthscales.value, rtol=0.3))
        self.assertTrue(np.allclose(var, m.var.value, rtol=0.3))


if __name__ == '__main__':
    unittest.main()
