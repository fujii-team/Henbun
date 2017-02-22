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

    def euclid_dist(self, X, X2):
        return np.sqrt(self.square_dist(X, X2)+1.0e-12)

    def Kdiag(self, X):
        if len(X.shape)==3: # batched case
            return np.ones((X.shape[0], X.shape[1]))
        else:
            return np.ones((X.shape[0]))

    def CsymKdiag(self,X):
        return np.diagonal(self.K(X, X), axis1=-2, axis2=-1)

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
        return RefStationary.CsymKdiag(self, X)

class RefMatern12(RefStationary):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        return np.exp(-self.euclid_dist(X, X2))

class RefCsymMatern12(RefMatern12):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        return RefMatern12.K(self, X, X2) + RefMatern12.K(self, X, -X2)

    def Kdiag(self, X):
        return RefStationary.CsymKdiag(self, X)

class RefMatern32(RefStationary):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        r = self.euclid_dist(X, X2)
        return (1. + np.sqrt(3.) * r) * np.exp(-np.sqrt(3.) * r)

class RefCsymMatern32(RefMatern32):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        return RefMatern32.K(self, X, X2) + RefMatern32.K(self, X, -X2)

    def Kdiag(self, X):
        return RefStationary.CsymKdiag(self, X)

class RefMatern52(RefStationary):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        r = self.euclid_dist(X, X2)
        return (1.0 + np.sqrt(5.) * r + 5. / 3. * np.square(r)) \
               * np.exp(-np.sqrt(5.) * r)

class RefCsymMatern52(RefMatern52):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def K(self, X, X2):
        return RefMatern52.K(self, X, X2) + RefMatern52.K(self, X, -X2)

    def Kdiag(self, X):
        return RefStationary.CsymKdiag(self, X)


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
        # Matern12
        self.m.k4 = hb.gp.kernels.UnitMatern12(lengthscales=self.l1)
        self.k4_ref = RefMatern12(self.l1)
        # CsymMatern12
        self.m.k5 = hb.gp.kernels.UnitCsymMatern12(lengthscales=self.l1)
        self.k5_ref = RefCsymMatern12(self.l1)
        # Matern32
        self.m.k6 = hb.gp.kernels.UnitMatern32(lengthscales=self.l1)
        self.k6_ref = RefMatern32(self.l1)
        # CsymMatern32
        self.m.k7 = hb.gp.kernels.UnitCsymMatern32(lengthscales=self.l1)
        self.k7_ref = RefCsymMatern32(self.l1)
        # Matern32
        self.m.k8 = hb.gp.kernels.UnitMatern52(lengthscales=self.l1)
        self.k8_ref = RefMatern52(self.l1)
        # CsymMatern32
        self.m.k9 = hb.gp.kernels.UnitCsymMatern52(lengthscales=self.l1)
        self.k9_ref = RefCsymMatern52(self.l1)
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
            K4 = self.m._session.run(self.m.k4.K(self.X))
            K5 = self.m._session.run(self.m.k5.K(self.X))
            K6 = self.m._session.run(self.m.k6.K(self.X))
            K7 = self.m._session.run(self.m.k7.K(self.X))
            K8 = self.m._session.run(self.m.k8.K(self.X))
            K9 = self.m._session.run(self.m.k9.K(self.X))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K4, self.k4_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K5, self.k5_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K6, self.k6_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K7, self.k7_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K8, self.k8_ref.K(self.X,self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K9, self.k9_ref.K(self.X,self.X), atol=1.0e-4))

    def test_K_batch(self):
        # batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X_batch))
            K2 = self.m._session.run(self.m.k2.K(self.X_batch))
            K3 = self.m._session.run(self.m.k3.K(self.X_batch))
            K4 = self.m._session.run(self.m.k4.K(self.X_batch))
            K5 = self.m._session.run(self.m.k5.K(self.X_batch))
            K6 = self.m._session.run(self.m.k6.K(self.X_batch))
            K7 = self.m._session.run(self.m.k7.K(self.X_batch))
            K8 = self.m._session.run(self.m.k8.K(self.X_batch))
            K9 = self.m._session.run(self.m.k9.K(self.X_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K4, self.k4_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K5, self.k5_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K6, self.k6_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K7, self.k7_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K8, self.k8_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K9, self.k9_ref.K(self.X_batch,self.X_batch), atol=1.0e-4))

    def test_K_batch_nonbatch(self):
        # make sure the batch and non-batch calculation are identical
        rng = np.random.RandomState(0)
        X = rng.randn(5,2)
        with self.m.tf_mode():
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k1.K(X)), # non-batch
                self.m._session.run(self.m.k1.K(X.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k2.K(X)), # non-batch
                self.m._session.run(self.m.k2.K(X.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k3.K(X)), # non-batch
                self.m._session.run(self.m.k3.K(X.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k4.K(X)), # non-batch
                self.m._session.run(self.m.k4.K(X.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k5.K(X)), # non-batch
                self.m._session.run(self.m.k5.K(X.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k6.K(X)), # non-batch
                self.m._session.run(self.m.k6.K(X.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k7.K(X)), # non-batch
                self.m._session.run(self.m.k7.K(X.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k8.K(X)), # non-batch
                self.m._session.run(self.m.k8.K(X.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k9.K(X)), # non-batch
                self.m._session.run(self.m.k9.K(X.reshape(1,-1,2)))))# batch

    def test_K2(self):
        # non-batch case
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X, self.X2))
            K2 = self.m._session.run(self.m.k2.K(self.X, self.X2))
            K3 = self.m._session.run(self.m.k3.K(self.X, self.X2))
            K4 = self.m._session.run(self.m.k4.K(self.X, self.X2))
            K5 = self.m._session.run(self.m.k5.K(self.X, self.X2))
            K6 = self.m._session.run(self.m.k6.K(self.X, self.X2))
            K7 = self.m._session.run(self.m.k7.K(self.X, self.X2))
            K8 = self.m._session.run(self.m.k8.K(self.X, self.X2))
            K9 = self.m._session.run(self.m.k9.K(self.X, self.X2))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K4, self.k4_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K5, self.k5_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K6, self.k6_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K7, self.k7_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K8, self.k8_ref.K(self.X,self.X2), atol=1.0e-4))
        self.assertTrue(np.allclose(K9, self.k9_ref.K(self.X,self.X2), atol=1.0e-4))
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
            K4 = self.m._session.run(self.m.k4.K(self.X_batch, self.X2_batch))
            K5 = self.m._session.run(self.m.k5.K(self.X_batch, self.X2_batch))
            K6 = self.m._session.run(self.m.k6.K(self.X_batch, self.X2_batch))
            K7 = self.m._session.run(self.m.k7.K(self.X_batch, self.X2_batch))
            K8 = self.m._session.run(self.m.k8.K(self.X_batch, self.X2_batch))
            K9 = self.m._session.run(self.m.k9.K(self.X_batch, self.X2_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K4, self.k4_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K5, self.k5_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K6, self.k6_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K7, self.k7_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K8, self.k8_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K9, self.k9_ref.K(self.X_batch,self.X2_batch), atol=1.0e-4))

    def test_K2_batch_nonbatch(self):
        # make sure the batch and non-batch calculation are identical
        rng = np.random.RandomState(0)
        X = rng.randn(5,2)
        X2 = rng.randn(6,2)
        with self.m.tf_mode():
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k1.K(X, X2)), # non-batch
                self.m._session.run(self.m.k1.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k2.K(X, X2)), # non-batch
                self.m._session.run(self.m.k2.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k3.K(X, X2)), # non-batch
                self.m._session.run(self.m.k3.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k4.K(X, X2)), # non-batch
                self.m._session.run(self.m.k4.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k5.K(X, X2)), # non-batch
                self.m._session.run(self.m.k5.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k6.K(X, X2)), # non-batch
                self.m._session.run(self.m.k6.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k7.K(X, X2)), # non-batch
                self.m._session.run(self.m.k7.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k8.K(X, X2)), # non-batch
                self.m._session.run(self.m.k8.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k9.K(X, X2)), # non-batch
                self.m._session.run(self.m.k9.K(X.reshape(1,-1,2), X2.reshape(1,-1,2)))))# batch

    def test_Kdiag(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.Kdiag(self.X))
            K2 = self.m._session.run(self.m.k2.Kdiag(self.X))
            K3 = self.m._session.run(self.m.k3.Kdiag(self.X))
            K4 = self.m._session.run(self.m.k4.Kdiag(self.X))
            K5 = self.m._session.run(self.m.k5.Kdiag(self.X))
            K6 = self.m._session.run(self.m.k6.Kdiag(self.X))
            K7 = self.m._session.run(self.m.k7.Kdiag(self.X))
            K8 = self.m._session.run(self.m.k8.Kdiag(self.X))
            K9 = self.m._session.run(self.m.k9.Kdiag(self.X))
        self.assertTrue(np.allclose(K1, self.k1_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K4, self.k4_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K5, self.k5_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K6, self.k6_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K7, self.k7_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K8, self.k8_ref.Kdiag(self.X), atol=1.0e-4))
        self.assertTrue(np.allclose(K9, self.k9_ref.Kdiag(self.X), atol=1.0e-4))

    def test_Kdiag_batch(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.Kdiag(self.X_batch))
            K2 = self.m._session.run(self.m.k2.Kdiag(self.X_batch))
            K3 = self.m._session.run(self.m.k3.Kdiag(self.X_batch))
            K4 = self.m._session.run(self.m.k4.Kdiag(self.X_batch))
            K5 = self.m._session.run(self.m.k5.Kdiag(self.X_batch))
            K6 = self.m._session.run(self.m.k6.Kdiag(self.X_batch))
            K7 = self.m._session.run(self.m.k7.Kdiag(self.X_batch))
            K8 = self.m._session.run(self.m.k8.Kdiag(self.X_batch))
            K9 = self.m._session.run(self.m.k9.Kdiag(self.X_batch))
        self.assertTrue(np.allclose(K1, self.k1_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K2, self.k2_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K3, self.k3_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K4, self.k4_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K5, self.k5_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K6, self.k6_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K7, self.k7_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K8, self.k8_ref.Kdiag(self.X_batch), atol=1.0e-4))
        self.assertTrue(np.allclose(K9, self.k9_ref.Kdiag(self.X_batch), atol=1.0e-4))

    def test_cholesky(self):
        with self.m.tf_mode():
            K1 = self.m._session.run(self.m.k1.K(self.X))
            K2 = self.m._session.run(self.m.k2.K(self.X))
            K3 = self.m._session.run(self.m.k3.K(self.X))
            K4 = self.m._session.run(self.m.k4.K(self.X))
            K5 = self.m._session.run(self.m.k5.K(self.X))
            K6 = self.m._session.run(self.m.k6.K(self.X))
            K7 = self.m._session.run(self.m.k7.K(self.X))
            K8 = self.m._session.run(self.m.k8.K(self.X))
            K9 = self.m._session.run(self.m.k9.K(self.X))
            chol1 = self.m._session.run(self.m.k1.Cholesky(self.X))
            chol2 = self.m._session.run(self.m.k2.Cholesky(self.X))
            chol3 = self.m._session.run(self.m.k3.Cholesky(self.X))
            chol4 = self.m._session.run(self.m.k4.Cholesky(self.X))
            chol5 = self.m._session.run(self.m.k5.Cholesky(self.X))
            chol6 = self.m._session.run(self.m.k6.Cholesky(self.X))
            chol7 = self.m._session.run(self.m.k7.Cholesky(self.X))
            chol8 = self.m._session.run(self.m.k8.Cholesky(self.X))
            chol9 = self.m._session.run(self.m.k9.Cholesky(self.X))
        # check shape
        self.assertTrue(np.allclose(chol1.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol2.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol3.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol4.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol5.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol6.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol7.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol8.shape, [self.X.shape[0], self.X.shape[0]]))
        self.assertTrue(np.allclose(chol9.shape, [self.X.shape[0], self.X.shape[0]]))

        self.assertTrue(np.allclose(K1, np.matmul(chol1, chol1.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K2, np.matmul(chol2, chol2.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K3, np.matmul(chol3, chol3.T), atol=9.0e-3))
        self.assertTrue(np.allclose(K4, np.matmul(chol4, chol4.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K5, np.matmul(chol5, chol5.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K6, np.matmul(chol6, chol6.T), atol=9.0e-3))
        self.assertTrue(np.allclose(K7, np.matmul(chol7, chol7.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K8, np.matmul(chol8, chol8.T), atol=9.0e-4))
        self.assertTrue(np.allclose(K9, np.matmul(chol9, chol9.T), atol=9.0e-4))
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
            K4 = self.m._session.run(self.m.k4.K(self.X_batch))
            K5 = self.m._session.run(self.m.k5.K(self.X_batch))
            K6 = self.m._session.run(self.m.k6.K(self.X_batch))
            K7 = self.m._session.run(self.m.k7.K(self.X_batch))
            K8 = self.m._session.run(self.m.k8.K(self.X_batch))
            K9 = self.m._session.run(self.m.k9.K(self.X_batch))

            chol1 = self.m._session.run(self.m.k1.Cholesky(self.X_batch))
            chol2 = self.m._session.run(self.m.k2.Cholesky(self.X_batch))
            chol3 = self.m._session.run(self.m.k3.Cholesky(self.X_batch))
            chol4 = self.m._session.run(self.m.k4.Cholesky(self.X_batch))
            chol5 = self.m._session.run(self.m.k5.Cholesky(self.X_batch))
            chol6 = self.m._session.run(self.m.k6.Cholesky(self.X_batch))
            chol7 = self.m._session.run(self.m.k7.Cholesky(self.X_batch))
            chol8 = self.m._session.run(self.m.k8.Cholesky(self.X_batch))
            chol9 = self.m._session.run(self.m.k9.Cholesky(self.X_batch))
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
            self.assertTrue(np.allclose(K4[i,...],
                                np.matmul(chol4[i,...], chol4[i,...].T), atol=1.0e-4))
            self.assertTrue(np.allclose(K5[i,...],
                                np.matmul(chol5[i,...], chol5[i,...].T), atol=1.0e-4))
            self.assertTrue(np.allclose(K6[i,...],
                                np.matmul(chol6[i,...], chol6[i,...].T), atol=1.0e-3))
            self.assertTrue(np.allclose(K7[i,...],
                                np.matmul(chol7[i,...], chol7[i,...].T), atol=1.0e-4))
            self.assertTrue(np.allclose(K8[i,...],
                                np.matmul(chol8[i,...], chol8[i,...].T), atol=1.0e-4))
            self.assertTrue(np.allclose(K9[i,...],
                                np.matmul(chol9[i,...], chol9[i,...].T), atol=1.0e-4))

    def test_cholesky_batch_nonbatch(self):
        # make sure the batch and non-batch calculation are identical
        rng = np.random.RandomState(0)
        X = rng.randn(5,2)
        with self.m.tf_mode():
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k1.K(X)),
                self.m._session.run(self.m.k1.K(X.reshape(1,-1,2)))))
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k2.K(X)),
                self.m._session.run(self.m.k2.K(X.reshape(1,-1,2)))))
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k3.K(X)),
                self.m._session.run(self.m.k3.K(X.reshape(1,-1,2)))))
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k4.K(X)),
                self.m._session.run(self.m.k4.K(X.reshape(1,-1,2)))))
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k5.K(X)),
                self.m._session.run(self.m.k5.K(X.reshape(1,-1,2)))))
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k6.K(X)),
                self.m._session.run(self.m.k6.K(X.reshape(1,-1,2)))))
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k7.K(X)),
                self.m._session.run(self.m.k7.K(X.reshape(1,-1,2)))))
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k8.K(X)),
                self.m._session.run(self.m.k8.K(X.reshape(1,-1,2)))))
            self.assertTrue(np.allclose(
                self.m._session.run(self.m.k9.K(X)),
                self.m._session.run(self.m.k9.K(X.reshape(1,-1,2)))))


if __name__ == '__main__':
    unittest.main()
