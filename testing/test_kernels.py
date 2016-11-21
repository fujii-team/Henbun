from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class RefStationary(object):
    def __init__(self, lengthscales):
        self.lengthscales = lengthscales

    def square_dist(self, X, X2):
        if len(X.shape)==3: # batched case
            dist = np.zeros((X.shape[0], X2.shape[0], X.shape[2]))
            l = self.lengthscales.reshape(-1,1)
        else: # non-batched case
            dist = np.zeros((X.shape[0], X2.shape[0]))
            l = self.lengthscales
        for i in range(X.shape[0]):
            for j in range(X2.shape[0]):
                x_dif = (X[i] - X2[j])/l
                dist[i,j] = np.sum(x_dif*x_dif, axis=0)
        return dist

    def Kdiag(self, X):
        if len(X.shape)==3: # batched case
            return np.ones((X.shape[0], X.shape[2]))
        else:
            return np.ones((X.shape[0]))

class RefRBF(RefStationary):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        return np.exp(-0.5*self.square_dist(X, X2))

class test_kernel_global(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.m = hb.model.Model()
        # scalar l1
        self.l1 = np.exp(rng.randn(1))
        self.m.k1 = hb.gp.kernels.UnitRBF(lengthscales=self.l1)
        self.k1_ref = RefRBF(self.l1)
        # vector lengthscales
        self.l2 = np.exp(rng.randn(2))
        self.m.k2 = hb.gp.kernels.UnitRBF(lengthscales=self.l2)
        self.k2_ref = RefRBF(self.l2)
        # non-batch X
        self.X = rng.randn(5,2)
        self.X2 = rng.randn(6,2)
        # batch_X
        self.X_batch = rng.randn(5,2,10)
        self.X2_batch = rng.randn(6,2,10)
        # initialize
        self.m.initialize()

    def test_K(self):
        # non-batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X))
            K2 = self.m._session.run(self.m.k2.K(self.X))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X,self.X), atol=1.0e-4))

    def test_K_batch(self):
        # batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X_batch))
            K2 = self.m._session.run(self.m.k2.K(self.X_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))

    def test_K2(self):
        # non-batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X, self.X2))
            K2 = self.m._session.run(self.m.k2.K(self.X, self.X2))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X,self.X2), atol=1.0e-4))
        # test if gradients works
        with self.m.tf_mode():
            grad = tf.gradients(K1, tf.trainable_variables())

    def test_K2_batch(self):
        # non-batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X_batch, self.X2_batch))
            K2 = self.m._session.run(self.m.k2.K(self.X_batch, self.X2_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))

    def test_Kdiag(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.Kdiag(self.X))
            K2 = self.m._session.run(self.m.k2.Kdiag(self.X))
        self.assertTrue(np.allclose(K1, self.k1_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.Kdiag(self.X), atol=1.0e-4))

    def test_Kdiag_batch(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.Kdiag(self.X_batch))
            K2 = self.m._session.run(self.m.k2.Kdiag(self.X_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.Kdiag(self.X_batch), atol=1.0e-4))

    def test_cholesky(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X))
            K2 = self.m._session.run(self.m.k2.K(self.X))
            chol1 = self.m._session.run(self.m.k1.Cholesky(self.X))
            chol2 = self.m._session.run(self.m.k2.Cholesky(self.X))
        # check shape
        self.assertTrue(np.allclose(chol1.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol2.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(K1, np.matmul(chol1, chol1.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K2, np.matmul(chol2, chol2.T), atol=9.0e-4))
        with self.m.tf_mode():
            grad = tf.gradients(chol1, tf.trainable_variables())

    def test_cholesky_batch(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X_batch))
            K2 = self.m._session.run(self.m.k2.K(self.X_batch))
            chol1 = self.m._session.run(self.m.k1.Cholesky(self.X_batch))
            chol2 = self.m._session.run(self.m.k2.Cholesky(self.X_batch))
        # check shape
        self.assertTrue(np.allclose(chol1.shape,
            [self.X_batch.shape[0], self.X_batch.shape[0], self.X_batch.shape[-1]]))
        self.assertTrue(np.allclose(chol2.shape,
            [self.X_batch.shape[0], self.X_batch.shape[0], self.X_batch.shape[-1]]))
        for i in range(self.X_batch.shape[-1]):
            self.assertTrue(np.allclose(K1[...,i],
                                np.matmul(chol1[...,i], chol1[...,i].T), atol=1.0e-4))
            self.assertTrue(np.allclose(K2[...,i],
                                np.matmul(chol2[...,i], chol2[...,i].T), atol=1.0e-4))

if __name__ == '__main__':
    unittest.main()