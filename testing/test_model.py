from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
import os

class SquareModel(hb.model.Model):
    def setUp(self):
        self.p = hb.param.Variable([2,3])

    @hb.model.AutoOptimize()
    def likelihood(self):
        return -tf.reduce_sum(tf.square(self.p))

class test_model(unittest.TestCase):
    def setUp(self):
        tf.set_random_seed(0)
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
        self.v = hb.variationals.Gaussian([2,3], collections = ['global2'])

    @hb.model.AutoOptimize()
    def likelihood(self):
        return -tf.reduce_sum(tf.square(self.p)) \
               -tf.reduce_sum(tf.square(self.q))

class test_model2(unittest.TestCase):
    def setUp(self):
        tf.set_random_seed(0)
        self.m = SquareModel2()
        self.filename = './saved_file.dat'
        self.files = [self.filename+'.data-00000-of-00001',
        self.filename+'.index', self.filename+'.meta', 'checkpoint']
        if os.path.exists(self.filename): os.remove(self.filename)
        [os.remove(f) for f in self.files if os.path.exists(f)]

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

    def test_save(self):
        self.m.q = np.ones((2,3))
        self.m.save(self.filename)
        # make sure the file is generated
        self.assertTrue(os.path.exists(self.files[0]))
        # make sure the model.restore()
        self.m2 = SquareModel2()
        self.m2.restore(self.filename)
        self.assertTrue(np.allclose(self.m2.q.value, np.ones((2,3))))
        self.assertFalse(np.allclose(self.m2.p.value, np.ones((2,3))))
        # make sure initialize() does not change the value
        self.m2.initialize()
        self.assertTrue(np.allclose(self.m2.q.value, np.ones((2,3))))
        self.setUp()

    def test_parameterized(self):
        """ make sure save method for parameterized works fine """
        self.m.v.q_mu = np.ones((2*3))
        self.m.v.save(self.filename)
        # make sure the file is generated
        self.assertTrue(os.path.exists(self.files[0]))
        # make sure the model.restore() works
        self.m2 = SquareModel2()
        self.m2.v.restore(self.filename)
        self.assertTrue(np.allclose(self.m2.v.q_mu.value, np.ones((2*3))))
        self.assertFalse(np.allclose(self.m2.p.value, np.ones((2,3))))
        # make sure initialize() does not change the value
        self.m2.initialize()
        self.assertTrue(np.allclose(self.m2.v.q_mu.value, np.ones((2*3))))
        self.setUp()

class MinibatchModel(hb.model.Model):
    def setUp(self):
        self.d = hb.param.MinibatchData(np.random.randn(100,2,3))
        self.p = hb.param.Variable([2,3])

    @hb.model.AutoOptimize()
    def likelihood(self):
        return -tf.reduce_sum(tf.square(self.p)) - self.KL()

class test_model3(unittest.TestCase):
    def test_minibatch(self):
        tf.set_random_seed(0)
        self.m = MinibatchModel()
        self.assertTrue(self.m._index.data_size is None)
        # compile
        self.m.likelihood().compile()
        self.assertTrue(self.m._index.data_size == 100)
        # change the data size
        self.m.d = hb.param.MinibatchData(np.random.randn(200,2,3))
        self.assertTrue(self.m._index.data_size == 100)
        # compile
        self.m.likelihood().compile()
        self.assertTrue(self.m._index.data_size == 200)

        # check the data size
        test_feed = self.m.test_feed_dict(20)
        with self.m.tf_mode():
            d = self.m.run(self.m.d, feed_dict = test_feed)
        self.assertTrue(d.shape == (20,2,3))

class ModelWithKeyword(hb.model.Model):
    def setUp(self, key1, key2):
        self.key1=key1
        self.key2=key2

class test_model4(unittest.TestCase):
    def test(self):
        model = ModelWithKeyword(key1='hoge', key2='foo')
        self.assertTrue(model.key1 == 'hoge')
        self.assertTrue(model.key2 == 'foo')


if __name__ == '__main__':
    unittest.main()
