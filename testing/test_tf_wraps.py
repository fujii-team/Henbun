from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class test_clip(unittest.TestCase):
    def test(self):
        """
        Make sure clip works.
        """
        rng = np.random.RandomState(0)
        custom_config = hb.settings.get_settings()
        custom_config.numerics.clip_by_value = False
        x = rng.randn(101,100).astype(np_float_type)
        with hb.settings.temp_settings(custom_config):
            m = hb.model.Model()
            m.nn = hb.nn.NeuralNet([100,99,98], neuron_types=tf.nn.relu,
                                    stddev=1.0)
            m.v = hb.variationals.Gaussian([100], stddev=100.0, mean=100.0)
            m.initialize()
            with m.tf_mode():
                y = m.run(m.nn(x))
                v = m.run(m.v)
        self.assertTrue(np.max(y) > 90)
        self.assertTrue(np.max(v) > 90)
        # --- clip enabled ---
        custom_config.numerics.clip_by_value = True
        with hb.settings.temp_settings(custom_config):
            m = hb.model.Model()
            m.nn = hb.nn.NeuralNet([100,99,98], neuron_types=tf.nn.relu,
                                    stddev=1.0)
            m.v = hb.variationals.Gaussian([100], stddev=100.0, mean=100.0)
            m.initialize()
            with m.tf_mode():
                y = m.run(m.nn(x))
                v = m.run(m.v)
        self.assertTrue(np.max(y) < 90)
        self.assertTrue(np.max(v) > 90)


class test_log_sum_exp(unittest.TestCase):
    def test(self):
        """
        Make sure log_sum_exp works.
        """
        rng = np.random.RandomState(0)
        a = rng.randn(2,3,4)
        b = rng.randn(2,3,4)
        c = rng.randn(2,3,4)

        with tf.Session() as sess:
            value = sess.run(
                 hb.tf_wraps.log_sum_exp(tf.pack([a, b, c], axis=1), axis=1))
        value_np = np.log(np.exp(a)+np.exp(b)+np.exp(c))
        self.assertTrue(np.allclose(value, value_np))


if __name__ == '__main__':
    unittest.main()
