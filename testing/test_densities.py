from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb

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

if __name__ == '__main__':
    unittest.main()
