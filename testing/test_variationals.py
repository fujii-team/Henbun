from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb

class test_variational(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.shapes = ['fullrank', 'diagonal']
        self.sqrts  = {'fullrank':np.exp(self.rng.randn(3,10,10)),
                       'diagonal':self.rng.randn(3,10)}
        for i in range(3): # remove upper triangular part
            for j in range(10):
                for k in range(j+1,10):
                    self.sqrts['fullrank'][i,j,k] = 0.
        self.x = self.rng.randn(3,10)
        self.m = {}
        for shape in self.shapes:
            self.m[shape] = hb.model.Model()
            self.m[shape].m = hb.variationals.Normal(self.x.shape[-1],
                                                n_layers=[3], q_shape=shape)
            self.m[shape].m.q_mu = self.x
            self.m[shape].m.q_sqrt = self.sqrts[shape]

        # immitate _draw_samples
        self.samples_iid = self.rng.randn(3,10)
        for shape in self.shapes:
            self.m[shape].u = self.samples_iid
            self.m[shape].m._sample()

        self.m[shape].initialize()


    def test_logdet(self):
        # true solution
        logdets = {'fullrank':np.zeros((3,10)), 'diagonal': np.zeros((3,10))}
        for i in range(3):
            for j in range(10):
                logdets['fullrank'][i,j] = 2.0*np.log(self.sqrts['fullrank'][i,j,j])
                logdets['diagonal'][i,j] = 2.0*self.sqrts['diagonal'][i,j]
        # check
        for shape in self.shapes:
            self.m[shape].initialize()
            with self.m[shape].tf_mode():
                logdet = self.m[shape]._session.run(self.m[shape].m.logdet)
            self.assertTrue(np.allclose(logdet, logdets[shape]))
    '''
    def test_project_samples(self):
        # true samples
        samples_post = {'fullrank': np.zeros((3,10,self.num_samples)),
                        'diagonal': np.zeros((3,10,self.num_samples))}
        shape = 'fullrank'
        for i in range(3):
            samples_post[shape][i,:,:] = self.x[i].reshape(-1,1) + \
                        np.dot(self.sqrts[shape][i], self.samples_iid[i])
        shape = 'diagonal'
        for i in range(3):
            for j in range(self.num_samples):
                samples_post[shape][i,:,j] = self.x[i] + \
                            self.sqrts[shape][i,:] * self.samples_iid[i,:,j]
        # check
        for shape in self.shapes:
            with self.m[shape].tf_mode():
                project_samples = self.m[shape]._session.run(
                        self.m[shape].m._project_samples(self.samples_iid))
            self.assertTrue(np.allclose(project_samples, samples_post[shape]))

    def test_variational_mode(self):
        for shape in self.shapes:
            with self.m[shape].tf_mode():
                 self.m[shape]._draw_samples(self.num_samples)
            samples = self.m[shape].m._samples_post
            with self.m[shape].tf_mode():
                with self.m[shape].variational_mode():
                    instance = self.m[shape].m
            self.assertTrue(samples is not None)
            self.assertTrue(samples is instance)




class test_variational_deep(unittest.TestCase):
    def setUp(self):

        pass


class test_with_variational(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.m = Henbun.param.Parameterized()
        self.m.m = Henbun.param.Parameterized()
        # local params
        self.local_p1 = rng.randn(10,2)
        self.m.local_p1 = Henbun.param.Variable(self.local_p1.copy(), vtype='local_param')
        # variational params
        self.variational_p1 = rng.randn(4,2)
        self.m.variational_p1 = Henbun.param.Variational(self.variational_p1)
    def test_variationals(self):
        vlist = self.m.sorted_variationals
        print(vlist)
        self.assertTrue(vlist[0] == self.m.variational_p1)


    '''
if __name__ == '__main__':
    unittest.main()
