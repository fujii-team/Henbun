from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb

class test_clip(unittest.TestCase):
    def test(self):
        """
        Make sure clip works.
        """
        rng = np.random.RandomState(0)
        custom_config = hb.settings.get_settings()
        custom_config.numerics.clip_by_value = False
        x = rng.randn(100,101)
        with hb.settings.temp_settings(custom_config):
            m = hb.model.Model()
            m.nn = hb.nn.NeuralNet([100,99,98], neuron_types=tf.nn.relu,
                                    stddev=1.0)
            m.initialize()
            with m.tf_mode():
                y = m.run(m.nn(x))
        self.assertTrue(np.max(y) > 90)
        # --- clip enabled ---
        custom_config.numerics.clip_by_value = True
        with hb.settings.temp_settings(custom_config):
            m = hb.model.Model()
            m.nn = hb.nn.NeuralNet([100,99,98], neuron_types=tf.nn.relu,
                                    stddev=1.0)
            m.initialize()
            with m.tf_mode():
                y = m.run(m.nn(x))
        self.assertTrue(np.max(y) < 90)



"""
In this test, we make sure n.n. model works fine.
"""
if __name__ == '__main__':
    unittest.main()
