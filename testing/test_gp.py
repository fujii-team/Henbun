from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32

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


class test_gp(unittest.TestCase):
    def test(self):
        m = gp()
        # run normal gpr
        m.likelihood_ana().compile(optimizer=tf.train.AdamOptimizer(0.01))
        m.likelihood_ana().optimize(maxiter=1000)
        lik = m.likelihood_ana().run()
        k_lengthscale = m.kern.lengthscales.value
        k_var = m.k_var.value
        var = m.var.value

        # run variational gpr
        m.kern.lengthscales=1.0
        m.k_var=1.0
        m.var  =1.0
        m.likelihood_var().compile(optimizer=tf.train.AdamOptimizer(0.01))
        m.likelihood_var().optimize(maxiter=10000)
        with m.tf_mode():
            [print(m.run(m.KL())) for _ in range(10)]

        # average samples for likelihood
        lik_var = np.mean([m.likelihood_var().run() for i in range(100)])
        print(lik, lik_var)
        print(k_lengthscale, m.kern.lengthscales.value)
        print(k_var, m.k_var.value)
        print(var, m.var.value)
        self.assertTrue(np.allclose(lik, m.likelihood_var().run(), atol=1.0))


if __name__ == '__main__':
    unittest.main()
