from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
from scipy.linalg import solve_triangular
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class test_einsum(unittest.TestCase):
    """
    Test variationals.einsum works fine
    """
    def test_matmul(self):
        v_global = hb.variationals.Normal((2,3),
            n_layers=[2,3], q_shape='fullrank')
        v_local = hb.variationals.Normal((2,3),
            n_layers=[2,3], q_shape='fullrank',collections=hb.param.graph_key.LOCAL)
        #         layer:ab+shape:cd
        einsum_global_ref = 'abcd,abd->abc'
        self.assertTrue(v_global._einsum_matmul() == einsum_global_ref)
        einsum_local_ref = 'abcde,abce->abcd'
        self.assertTrue(v_local._einsum_matmul() == einsum_local_ref)

class test_variational(unittest.TestCase):
    """
    Test variationals works fine
    """
    def setUp(self):
        """ Prepare fullrank and diagonal variationals """
        tf.set_random_seed(0)
        self.rng = np.random.RandomState(0)
        self.shapes = ['fullrank', 'diagonal']
        self.sqrts  = {'fullrank':self.rng.randn(3,10,10)*0.5,
                       'diagonal':self.rng.randn(3,10)*0.5-0.5}
        for i in range(3): # remove upper triangular part
            for j in range(10):
                self.sqrts['fullrank'][i,j,j] = np.exp(self.sqrts['fullrank'][i,j,j])
                for k in range(j+1,10):
                    self.sqrts['fullrank'][i,j,k] = 0.
        self.x = self.rng.randn(3,10)*0.3
        self.m = {}
        for shape in self.shapes:
            self.m[shape] = hb.model.Model()
            self.m[shape].m = hb.variationals.Normal(self.x.shape[-1],
                                                n_layers=[3], q_shape=shape)
            self.m[shape].m.q_mu = self.x
            self.m[shape].m.q_sqrt = self.sqrts[shape]

        # immitate _draw_samples
        self.samples_iid = self.rng.randn(3,10).astype(np_float_type)
        for shape in self.shapes:
            self.m[shape].initialize()

    def test_get_variables(self):
        """ Test get_variables certainly works even if in tf_mode """
        for shape in self.shapes:
            with self.m[shape].tf_mode():
                variables = self.m[shape].get_variables()
            self.assertTrue(self.m[shape].m.q_mu in variables)
            self.assertTrue(self.m[shape].m.q_sqrt in variables)

    def test_parent(self):
        """ make sure its parent is model """
        for shape in self.shapes:
            self.assertTrue(self.m[shape].m._parent is self.m[shape])

    def test_logdet(self):
        """ make sure logdet is calculated correctly """
        # true solution
        logdets = {'fullrank':np.zeros((3,10)), 'diagonal': np.zeros((3,10))}
        for i in range(3):
            for j in range(10):
                logdets['fullrank'][i,j] = 2.0*np.log(self.sqrts['fullrank'][i,j,j])
                logdets['diagonal'][i,j] = 2.0*self.sqrts['diagonal'][i,j]
        # check
        for shape in self.shapes:
            self.m[shape].initialize()
            m = self.m[shape].m
            with self.m[shape].tf_mode():
                logdet = self.m[shape]._session.run(m.logdet)
            self.assertTrue(np.allclose(logdet, logdets[shape]))

    def test_project_samples(self):
        """ make sure project_samples is calculated correctly """
        # true samples
        samples_post = {'fullrank': np.zeros((3,10)),
                        'diagonal': np.zeros((3,10))}
        shape = 'fullrank'
        for i in range(3):
            samples_post[shape][i,:] = self.x[i] + \
                        np.dot(self.sqrts[shape][i], self.samples_iid[i])
        shape = 'diagonal'
        for i in range(3):
            samples_post[shape][i,:] = self.x[i] + \
                        np.exp(self.sqrts[shape][i,:]) * self.samples_iid[i,:]
        # check
        for shape in self.shapes:
            m = self.m[shape].m
            with self.m[shape].tf_mode():
                project_samples = self.m[shape]._session.run(
                        m._sample(tf.convert_to_tensor(self.samples_iid)))
            self.assertTrue(np.allclose(project_samples, samples_post[shape]))
            # check if _tensor is created
            self.assertTrue(isinstance(self.m[shape].m._tensor, tf.Tensor))

    def test_KL(self):
        """ Compare with analytical KL """
        for shape in self.shapes:
            KL = 0.0
            num_samples=100
            with self.m[shape].tf_mode():
                for i in range(num_samples):
                    KL += self.m[shape].run(self.m[shape].KL())
            KL = KL/num_samples
            KL_ana = gaussian_KL(self.m[shape].m.q_mu.value,
                                 self.m[shape].m.q_sqrt.value, q_shape=shape)
            print(KL, KL_ana)
            self.assertTrue(np.allclose(KL, KL_ana, rtol=0.1))

    def test_feed(self):
        """ Make sure feed method does nothing for Global case"""
        for shape in self.shapes:
            tensor_ref = self.m[shape].m._tensor
            self.m[shape].m.feed(np.ones(10))
            self.assertTrue(tensor_ref is self.m[shape].m._tensor)

class test_variational_local(unittest.TestCase):
    def setUp(self):
        tf.set_random_seed(0)
        self.rng = np.random.RandomState(0)
        self.shapes = ['fullrank', 'diagonal']
        # n_layers=[3], n_batch=2, shape=[10,10] or [10]
        self.sqrts  = {'fullrank':self.rng.randn(3,2,10,10).astype(dtype=np_float_type),
                       'diagonal':self.rng.randn(3,2,10).astype(dtype=np_float_type)}
        for i in range(3): # remove upper triangular part
            for j in range(10):
                self.sqrts['fullrank'][i,:,j,j] = np.exp(0.1*self.sqrts['fullrank'][i,:,j,j])
                for k in range(j+1,10):
                    self.sqrts['fullrank'][i,:,j,k] = 0.
        self.x = self.rng.randn(3,2,10).astype(np_float_type)
        self.m = {}
        for shape in self.shapes:
            self.m[shape] = hb.model.Model()
            # Local Variational variable
            self.m[shape].m = hb.variationals.Normal([10],
                n_layers=[3], q_shape=shape, collections=hb.param.graph_key.LOCAL)
            if shape is 'fullrank':
                self.m[shape].m.q_mu.feed(self.x)
                self.m[shape].m.q_sqrt.feed(self.sqrts[shape].reshape(3,2,100))
            else:
                self.m[shape].m.q_mu.feed(self.x)
                self.m[shape].m.q_sqrt.feed(self.sqrts[shape])
            # batched Global param
            self.m[shape].q = hb.variationals.Normal([10], n_layers=[3],
                                                    n_batch=2, q_shape=shape)
            self.m[shape].q.q_mu = self.x
            self.m[shape].q.q_sqrt = self.sqrts[shape]
            self.m[shape].initialize()
        # immitate _draw_samples
        self.samples_iid = self.rng.randn(3,2,10).astype(np_float_type)

    def test_get_variables(self):
        """ Test get_variables certainly works even if in tf_mode """
        for shape in self.shapes:
            with self.m[shape].tf_mode():
                variables = self.m[shape].get_variables(hb.param.graph_key.LOCAL)
                feed_size = self.m[shape].feed_size
                # test feed certainly works
                self.m[shape].feed(self.rng.randn(3, 100, feed_size).astype(np_float_type))
            # check feed_size is the same in tf_mode
            self.assertTrue(feed_size == self.m[shape].feed_size)
            self.assertTrue(self.m[shape].m.q_mu in variables)
            self.assertTrue(self.m[shape].m.q_sqrt in variables)
            # check certainly variational.feed works
            self.assertTrue(hasattr(self.m[shape].m, '_tensor'))

    def test_logdet(self):
        # true solution
        logdets = {'fullrank':np.zeros((3,2,10)), 'diagonal': np.zeros((3,2,10))}
        for i in range(3):
            for j in range(2):
                for k in range(10):
                    logdets['fullrank'][i,j,k] = 2.0*np.log(self.sqrts['fullrank'][i,j,k,k])
                    logdets['diagonal'][i,j,k] = 2.0*self.sqrts['diagonal'][i,j,k]
        # check
        for shape in self.shapes:
            with self.m[shape].m.tf_mode():
                logdet_m = self.m[shape].run(self.m[shape].m.logdet)
            with self.m[shape].q.tf_mode():
                logdet_q = self.m[shape].run(self.m[shape].q.logdet)
            self.assertTrue(np.allclose(logdet_m, logdets[shape]))
            self.assertTrue(np.allclose(logdet_q, logdets[shape]))

    def test_tf_mode(self):
        for shape in self.shapes:
            with self.m[shape].tf_mode():
                if shape is 'diagonal':
                    self.m[shape].m =\
                            np.concatenate([self.x, self.sqrts[shape].reshape(3,2,10)], axis=2)
                else:
                    self.m[shape].m =\
                            np.concatenate([self.x, self.sqrts[shape].reshape(3,2,100)], axis=2)
                # make sure Variational is seen as tf.Tensor in tf_mode
                self.assertTrue(isinstance(self.m[shape].m, tf.Tensor))
            # make sure Variational is seen as Variational in non-tf_mode
            self.assertTrue(isinstance(self.m[shape].m, hb.variationals.Variational))

class VariationalModel(hb.model.Model):
    def setUp(self):
        tf.set_random_seed(0)
        self.q_global = hb.variationals.Normal(shape=[3])
        self.q_local = hb.variationals.Normal(shape=[3], collections=hb.param.graph_key.LOCAL)
        self.x = hb.param.Variable(shape=[10,6])
    @hb.model.AutoOptimize()
    def likelihood_global(self):
        self.q_local = self.x
        return -tf.reduce_sum(tf.square(self.q_global))
    @hb.model.AutoOptimize()
    def likelihood_local(self):
        self.q_local = self.x
        return -tf.reduce_sum(tf.square(self.q_local))


class test_variational_model(unittest.TestCase):
    def setUp(self):
        tf.set_random_seed(0)
        self.m = VariationalModel()

    def test_compile_global(self):
        self.m.likelihood_global().compile()

    def test_compile_local(self):
        self.m.likelihood_local().compile()

    def test_sorted_variables(self):
        self.assertTrue(len(self.m.sorted_variables) == 3)

    def test_KL(self):
        self.m.initialize()
        with self.m.tf_mode():
            self.m.q_local = self.m.x
            KL = self.m.KL()
        kl = self.m.run(KL)
        self.assertFalse(np.allclose(kl, 0))


class SimplestVariationalModel(hb.model.Model):
    def setUp(self):
        self.q = hb.variationals.Normal(shape=[3,2], q_shape='fullrank')

class test_initial_values(unittest.TestCase):
    def test(self):
        """ make sure all the components in the full-rank diagonal is positive """
        m = SimplestVariationalModel()
        m.initialize()
        sqrt = m.q.q_sqrt.value
        diag = np.diagonal(sqrt)
        self.assertTrue(np.all(diag > 0.0))

class TestGaussian(unittest.TestCase):
    """
    Test several initial values work with no error
    """
    def test_several_inits(self):
        tf.set_random_seed(0)
        m = hb.model.Model()
        m.g1 = hb.variationals.Gaussian(shape=[3,2], n_layers=[1,2], n_batch=0,
                mean= 1.0, stddev=0.5, scale_shape=[3,2], scale_n_layers=[1,2])
        m.g2 = hb.variationals.Gaussian(shape=[3,2], n_layers=[1,2], n_batch=0,
                mean=-1.0, stddev=0.5, scale_shape=[3,2], scale_n_layers=[1,2])
        m.g3 = hb.variationals.Gaussian(shape=[3,2], n_layers=[1,2], n_batch=0,
                mean= 0.0, stddev=1.0, scale_shape=[3,2], scale_n_layers=[1,2])

        with m.tf_mode():
            g1 = m.run(m.g1)
            g2 = m.run(m.g2)
            g3 = m.run(m.g3)

class TestBeta(unittest.TestCase):
    """
    Test several initial values work with no error
    """
    def test_several_inits(self):
        tf.set_random_seed(0)
        m = hb.model.Model()
        m.g1 = hb.variationals.Beta(shape=[3,2], n_layers=[1,2], n_batch=0,
                mean= 1.0, stddev=0.5, scale_shape=[3,2], scale_n_layers=[1,2])
        m.g2 = hb.variationals.Beta(shape=[3,2], n_layers=[1,2], n_batch=0,
                mean=-1.0, stddev=0.5, scale_shape=[3,2], scale_n_layers=[1,2])
        m.g3 = hb.variationals.Beta(shape=[3,2], n_layers=[1,2], n_batch=0,
                mean= 0.0, stddev=1.0, scale_shape=[3,2], scale_n_layers=[1,2])

        with m.tf_mode():
            g1 = m.run(m.g1)
            g2 = m.run(m.g2)
            g3 = m.run(m.g3)

def gaussian_KL(mu, L, q_shape='diagonal'):
    """
    Estimate analytical KL[p||q]
    where p = N(mu,L L^T) and q = N(0,I)
    arg:
    - mu: mean vector. np.array sized [N,n]
    - L : cholesky matrix. np.array sized [N,n,n],
    """
    KL = 0.0
    for i in range(mu.shape[0]):
        n = mu.shape[1]
        mu1 = mu[i,:]
        if q_shape is 'diagonal':
            L1  = L[i,:]
            logdet = 2.0*np.sum(L1)
            trace = np.sum(np.exp(2.0*L1))
        else: # fullrank
            L1  = L[i,:,:]
            logdet = np.sum(np.log(np.diagonal(L1)))
            trace = np.sum(np.square(L1))
        KL += -logdet - n + trace + np.dot(mu1.T, mu1)
    return KL*0.5

class TestBeta(unittest.TestCase):
    """ Just make sure it works without no error """
    def test(self):
        self.m = hb.model.Model()
        self.m.v = hb.variationals.Beta(shape=[3,2], n_layers=[3], n_batch=2)
        self.m.initialize()
        with self.m.tf_mode():
            self.m.run(self.m.KL())

if __name__ == '__main__':
    unittest.main()
