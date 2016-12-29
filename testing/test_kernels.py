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
            dist = np.zeros((X.shape[0], X.shape[1], X2.shape[1]))
            for i in range(X.shape[1]):
                for j in range(X2.shape[1]):
                    x_dif = (X[:,i,:] - X2[:,j,:])/self.lengthscales # [N,d]
                    dist[:,i,j] = np.sum(x_dif*x_dif, axis=-1)
        else: # non-batched case
            dist = np.zeros((X.shape[0], X2.shape[0]))
            for i in range(X.shape[0]):
                for j in range(X2.shape[0]):
                    x_dif = (X[i] - X2[j])/self.lengthscales
                    dist[i,j] = np.sum(x_dif*x_dif, axis=-1)
        return dist

    def Kdiag(self, X):
        if len(X.shape)==3: # batched case
            return np.ones((X.shape[0], X.shape[1]))
        else:
            return np.ones((X.shape[0]))

class RefRBF(RefStationary):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        return np.exp(-0.5*self.square_dist(X, X2))

class RefCsymRBF(RefStationary):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        return np.exp(-0.5*self.square_dist(X,  X2)) \
              +np.exp(-0.5*self.square_dist(X, -X2))

    def Kdiag(self, X):
        if len(X.shape)==3: # batched case
            l = self.lengthscales.reshape(-1,1)
            Xt = np.sum(X/l*X/l, axis=2)
            diag = np.ones([X.shape[0], X.shape[1]])
            for i in range(X.shape[1]):
                diag[:,i] += np.exp(-2.0*Xt[:,i])
        else:
            l = self.lengthscales
            Xt = np.sum(X/l*X/l, axis=1)
            diag = np.ones([X.shape[0]])
            for i in range(X.shape[0]):
                diag[i] += np.exp(-2.0*Xt[i])
        return diag


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
        # csym RBF
        self.m.k3 = hb.gp.kernels.UnitCsymRBF(lengthscales=self.l1)
        self.k3_ref = RefCsymRBF(self.l1)
        # non-batch X
        self.X = rng.randn(5,2)
        self.X2 = rng.randn(6,2)
        # batch_X
        self.X_batch = rng.randn(10,5,2)
        self.X2_batch = rng.randn(10,6,2)
        # initialize
        self.m.initialize()

    def test_K(self):
        # non-batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X))
            K2 = self.m._session.run(self.m.k2.K(self.X))
            K3 = self.m._session.run(self.m.k3.K(self.X))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.K(self.X,self.X), atol=1.0e-4))

    def test_K_batch(self):
        # batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X_batch))
            K2 = self.m._session.run(self.m.k2.K(self.X_batch))
            K3 = self.m._session.run(self.m.k3.K(self.X_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))

    def test_K2(self):
        # non-batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X, self.X2))
            K2 = self.m._session.run(self.m.k2.K(self.X, self.X2))
            K3 = self.m._session.run(self.m.k3.K(self.X, self.X2))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.K(self.X,self.X2), atol=1.0e-4))
        # test if gradients works
        with self.m.tf_mode():
            loss = tf.reduce_sum(self.m.k2.K(self.X, self.X2))
            grad = tf.gradients(loss, self.m.get_tf_variables())
        grads = [self.m.run(g) for g in grad if g is not None]
        self.assertTrue(len(grad)>0)

    def test_K2_batch(self):
        # non-batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X_batch, self.X2_batch))
            K2 = self.m._session.run(self.m.k2.K(self.X_batch, self.X2_batch))
            K3 = self.m._session.run(self.m.k3.K(self.X_batch, self.X2_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))

    def test_Kdiag(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.Kdiag(self.X))
            K2 = self.m._session.run(self.m.k2.Kdiag(self.X))
            K3 = self.m._session.run(self.m.k3.Kdiag(self.X))
        self.assertTrue(np.allclose(K1, self.k1_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.Kdiag(self.X), atol=1.0e-4))

    def test_Kdiag_batch(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.Kdiag(self.X_batch))
            K2 = self.m._session.run(self.m.k2.Kdiag(self.X_batch))
            K3 = self.m._session.run(self.m.k3.Kdiag(self.X_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.Kdiag(self.X_batch), atol=1.0e-4))

    def test_cholesky(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X))
            K2 = self.m._session.run(self.m.k2.K(self.X))
            K3 = self.m._session.run(self.m.k3.K(self.X))
            chol1 = self.m._session.run(self.m.k1.Cholesky(self.X))
            chol2 = self.m._session.run(self.m.k2.Cholesky(self.X))
            chol3 = self.m._session.run(self.m.k3.Cholesky(self.X))
        # check shape
        self.assertTrue(np.allclose(chol1.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol2.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol3.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(K1, np.matmul(chol1, chol1.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K2, np.matmul(chol2, chol2.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K3, np.matmul(chol3, chol3.T), atol=9.0e-3))
        with self.m.tf_mode():
            loss = tf.reduce_sum(self.m.k1.Cholesky(self.X))
            grad = tf.gradients(loss, self.m.get_tf_variables())
        grads = [self.m.run(g) for g in grad if g is not None]
        self.assertTrue(len(grad)>0)

    def test_cholesky_batch(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X_batch))
            K2 = self.m._session.run(self.m.k2.K(self.X_batch))
            K3 = self.m._session.run(self.m.k3.K(self.X_batch))
            chol1 = self.m._session.run(self.m.k1.Cholesky(self.X_batch))
            chol2 = self.m._session.run(self.m.k2.Cholesky(self.X_batch))
            chol3 = self.m._session.run(self.m.k3.Cholesky(self.X_batch))
        # check shape
        self.assertTrue(np.allclose(chol1.shape,
            [self.X_batch.shape[0], self.X_batch.shape[1], self.X_batch.shape[1]]))
        self.assertTrue(np.allclose(chol2.shape,
            [self.X_batch.shape[0], self.X_batch.shape[1], self.X_batch.shape[1]]))
        self.assertTrue(np.allclose(chol3.shape,
            [self.X_batch.shape[0], self.X_batch.shape[1], self.X_batch.shape[1]]))
        for i in range(self.X_batch.shape[0]):
            self.assertTrue(np.allclose(K1[i,...],
                                np.matmul(chol1[i,...], chol1[i,...].T), atol=1.0e-4))
            self.assertTrue(np.allclose(K2[i,...],
                                np.matmul(chol2[i,...], chol2[i,...].T), atol=1.0e-4))
            self.assertTrue(np.allclose(K3[i,...],
                                np.matmul(chol3[i,...], chol3[i,...].T), atol=1.0e-3))

if __name__ == '__main__':
    unittest.main()
