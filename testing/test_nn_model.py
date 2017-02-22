from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb

class SquareModel(hb.model.Model):
    def setUp(self):
        self.p = hb.param.Variable([2,3])

    def likelihood(self):
        return -tf.reduce_sum(tf.square(self.p))

class test_square(unittest.TestCase):
    def setUp(self):
        tf.set_random_seed(0)
        self.m = SquareModel()

    def test_manual_optimize(self):
        trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.m.initialize()
        with self.m.tf_mode():
            op = tf.negative(self.m.likelihood())
            opt_op = trainer.minimize(op, var_list = self.m.get_tf_variables())
            self.m._session.run(tf.variables_initializer(tf.global_variables()))
        for i in range(1000):
            self.m._session.run(opt_op)
        self.assertTrue(np.allclose(self.m.p.value, np.zeros((2,3)), atol=1.0e-4))

"""
In this test, we make sure n.n. model works fine.
"""
if __name__ == '__main__':
    unittest.main()
