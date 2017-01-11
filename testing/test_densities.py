from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from scipy.special import loggamma
from Henbun._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class test_bimixture(unittest.TestCase):
    def test(self):
        rng = np.random.RandomState(0)
        a = rng.randn(2,3,4)
        b = rng.randn(2,3,4)
        frac = rng.uniform(size=(2,1,1))

        with tf.Session() as sess:
            logp0 = sess.run(hb.densities.gaussian(a.astype(np.float32), 0.0, 2.0))
            logp1 = sess.run(hb.densities.student_t(b.astype(np.float32), 0.0, 2.0, 3.0))
            logp_mix = sess.run(hb.densities.bimixture(frac.astype(np.float32), logp0, logp1))

        logp_np = np.log(frac*np.exp(logp0)+(1-frac)*np.exp(logp1))
        self.assertTrue(np.allclose(logp_mix, logp_np))

def student_t_ref(x, mu, scale, nu):
    """ numpy method for student t log density """
    const = loggamma(0.5*(nu+1.0)) - loggamma(0.5*nu) \
         - 0.5*(np.log(scale*scale) + np.log(nu) + np.log(np.pi))

    return const - 0.5*(nu + 1.) *\
        np.log(1.0 + (1.0/nu)*((x-mu)/scale)**2.0)

class test_student_t(unittest.TestCase):
    def test_np(self):
        """ With numpy array parameter """
        rng = np.random.RandomState(0)
        x = rng.randn(2,3,4).astype(np_float_type)
        mu = rng.randn(2,3,4).astype(np_float_type)
        scale = np.exp(rng.randn(2,3,4).astype(np_float_type))
        nu = np.exp(rng.randn(2,3,4).astype(np_float_type))

        # scalar deg_free
        logp_ref = student_t_ref(x, mu, scale, 3.0)
        with tf.Session() as sess:
            logp = sess.run(hb.densities.student_t(x, mu, scale, 3.0))
        self.assertTrue(logp.dtype == 'float32')
        self.assertTrue(np.allclose(logp, logp_ref, atol=1.0e-5))

        # tensor_input
        with tf.Session() as sess:
            logp = sess.run(hb.densities.student_t(
                tf.constant(x, float_type),
                tf.constant(mu, float_type),
                tf.constant(scale, float_type),
                3.0))
        self.assertTrue(logp.dtype == 'float32')
        self.assertTrue(np.allclose(logp, logp_ref, atol=1.0e-5))

        # tensor deg_free
        logp_ref = student_t_ref(x, mu, scale, nu)
        with tf.Session() as sess:
            logp = sess.run(hb.densities.student_t(x, mu, scale, nu))
        self.assertTrue(logp.dtype == 'float32')
        self.assertTrue(np.allclose(logp, logp_ref, atol=1.0e-5))

        # tensor_input
        with tf.Session() as sess:
            logp = sess.run(hb.densities.student_t(
                tf.constant(x, float_type),
                tf.constant(mu, float_type),
                tf.constant(scale, float_type),
                tf.constant(nu, float_type)))
        self.assertTrue(logp.dtype == 'float32')
        self.assertTrue(np.allclose(logp, logp_ref, atol=1.0e-5))


if __name__ == '__main__':
    unittest.main()
