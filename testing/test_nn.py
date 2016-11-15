from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb

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
        x = rng.randn(5,3,6)
        with m.tf_mode():
            y_nn_op = m.nn(x)
            y_manual_op =  b2 + tf.batch_matmul(w2,
                tf.sigmoid(b1 + tf.batch_matmul(w1, x)))
            y_nn = m._session.run(y_nn_op)
            y_manual = m._session.run(y_manual_op)
        self.assertTrue(np.allclose(y_nn, y_manual))

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
        x = rng.randn(6,5,3,6)
        with m.tf_mode():
            y_nn_op = m.nn(x)
            y_manual_op =  b3 + tf.batch_matmul(w3,
                tf.nn.relu(b2 + tf.batch_matmul(w2,
             tf.nn.sigmoid(b1 + tf.batch_matmul(w1, x)))))
            y_nn = m._session.run(y_nn_op)
            y_manual = m._session.run(y_manual_op)
        self.assertTrue(np.allclose(y_nn, y_manual, atol=1.e-7))


if __name__ == '__main__':
    unittest.main()
