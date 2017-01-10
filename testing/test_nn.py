from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
float_type = settings.dtypes.float_type

class test_nn(unittest.TestCase):

    def test1(self):
        rng = np.random.RandomState(0)
        m = hb.model.Model()
        m.nn = hb.nn.NeuralNet([3,2,4], n_layers=[5], neuron_types=tf.sigmoid)
        m.initialize()
        # manual construction
        w1 = m.nn.matbias0.w.value
        b1 = m.nn.matbias0.b.value
        w2 = m.nn.matbias1.w.value
        b2 = m.nn.matbias1.b.value
        # test
        x = tf.constant(rng.randn(5,6,3), float_type)
        with m.tf_mode():
            y_nn_op = m.nn(x)
            y_manual_op = tf.batch_matmul(tf.sigmoid(tf.batch_matmul(x,w1)+b1), w2)+b2
            y_nn = m._session.run(y_nn_op)
            y_manual = m._session.run(y_manual_op)
        self.assertTrue(np.allclose(y_nn, y_manual, atol=1.0e-4))

    def test2(self):
        rng = np.random.RandomState(0)
        m = hb.model.Model()
        m.nn = hb.nn.NeuralNet([3,2,4,5], n_layers=[6,5], neuron_types=[tf.nn.sigmoid, tf.nn.relu])
        m.initialize()
        # manual construction
        w1 = m.nn.matbias0.w.value
        b1 = m.nn.matbias0.b.value
        w2 = m.nn.matbias1.w.value
        b2 = m.nn.matbias1.b.value
        w3 = m.nn.matbias2.w.value
        b3 = m.nn.matbias2.b.value
        # test
        x = tf.constant(rng.randn(6,5,6,3), float_type)
        with m.tf_mode():
            y_nn_op = m.nn(x)
            y_manual_op = tf.batch_matmul(
                tf.nn.relu(tf.batch_matmul(
                tf.nn.sigmoid(tf.batch_matmul(
                    x,w1)+b1), w2)+b2), w3)+b3
            y_nn = m._session.run(y_nn_op)
            y_manual = m._session.run(y_manual_op)
        self.assertTrue(np.allclose(y_nn, y_manual, atol=1.e-4))


if __name__ == '__main__':
    unittest.main()
