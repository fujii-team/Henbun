from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb

class SquareModel(hb.model.Model):
    def setUp(self):
        self.p = hb.param.Variable([2,3])

    @hb.model.AutoOptimize()
    def likelihood(self):
        return -tf.reduce_sum(tf.square(self.p))

class test_model(unittest.TestCase):
    def setUp(self):
        self.m = SquareModel()

    def test_auto_optimize(self):
        optimizer = self.m.likelihood()
        optimizer.compile(tf.train.AdamOptimizer(learning_rate=0.01))
        # just run
        val = optimizer.run()
        self.assertTrue(np.allclose(val, -np.sum(np.square(self.m.p.value))))
        # optimize
        optimizer.optimize(maxiter=1500)
        self.assertTrue(np.allclose(self.m.p.value, np.zeros((2,3)), atol=1.0e-4))


class SquareModel2(hb.model.Model):
    def setUp(self):
        self.p = hb.param.Variable([2,3], collections = ['global1', 'global2'])
        self.q = hb.param.Variable([2,3], collections = ['global2'])

    @hb.model.AutoOptimize()
    def likelihood(self):
        return -tf.reduce_sum(tf.square(self.p)) \
               -tf.reduce_sum(tf.square(self.q))

class test_model2(unittest.TestCase):
    def setUp(self):
        self.m = SquareModel2()

    def test_initialize_finalize(self):
        optimizer = self.m.likelihood()
        self.assertTrue(self.m.p._assigned)
        optimizer.compile(tf.train.AdamOptimizer(learning_rate=0.01), collection='global1')
        self.assertFalse(self.m.p._assigned)
        optimizer.optimize()
        self.assertFalse(self.m.p._assigned)

    def test_auto_optimize(self):
        optimizer = self.m.likelihood()
        # optimize p only
        optimizer.compile(tf.train.AdamOptimizer(learning_rate=0.01), collection='global1')
        optimizer.optimize(maxiter=1500)
        self.assertTrue(np.allclose(self.m.p.value, np.zeros((2,3)), atol=1.0e-4))
        self.assertFalse(np.allclose(self.m.q.value, np.zeros((2,3)), atol=1.0e-4))
        # optimize p and q
        optimizer.compile(tf.train.AdamOptimizer(learning_rate=0.01), collection='global2')
        # check self.p value does not change after the compilation
        self.assertTrue(np.allclose(self.m.p.value, np.zeros((2,3)), atol=1.0e-3))
        optimizer.optimize(maxiter=1000)
        self.assertTrue(np.allclose(self.m.p.value, np.zeros((2,3)), atol=1.0e-3))
        self.assertTrue(np.allclose(self.m.q.value, np.zeros((2,3)), atol=1.0e-3))

class MinibatchModel(hb.model.Model):
    def setUp(self):
        self.d = hb.param.MinibatchData(np.random.randn(2,3,100))
        self.p = hb.param.Variable([2,3])

    @hb.model.AutoOptimize()
    def likelihood(self):
        return -tf.reduce_sum(tf.square(self.p))

class test_model3(unittest.TestCase):
    def test_minibatch(self):
        self.m = MinibatchModel()
        self.assertTrue(self.m._index.data_size is None)
        # compile
        self.m.likelihood().compile()
        self.assertTrue(self.m._index.data_size == 100)
        # change the data size
        self.m.d = hb.param.MinibatchData(np.random.randn(2,3,200))
        self.assertTrue(self.m._index.data_size == 100)
        # compile
        self.m.likelihood().compile()
        self.assertTrue(self.m._index.data_size == 200)



if __name__ == '__main__':
    unittest.main()
