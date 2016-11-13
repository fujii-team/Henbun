from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
from Henbun.param import graph_key
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

# got from GPflow
class NamingTests(unittest.TestCase):
    def test_unnamed(self):
        p = hb.param.Variable([1])
        self.assertTrue(p.name == 'unnamed')

    def test_bad_parent(self):
        p = hb.param.Variable([1])
        m = hb.param.Parameterized()
        p._parent = m  # do not do this.
        with self.assertRaises(ValueError):
            print(p.name)

    def test_two_parents(self):
        m = hb.param.Parameterized()
        m.p = hb.param.Variable([1])
        m.p2 = m.p  # do not do this!
        with self.assertRaises(ValueError):
            print(m.p.name)

class ParamTestsScalar(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        tf.reset_default_graph()
        self.m = hb.model.Model()
        self.m.p = hb.param.Variable([1])
        self.m.q = hb.param.Variable([3], collections=graph_key.LOCAL)
        self.m.r = hb.param.Variable([5,4], collections=graph_key.LOCAL)

    def test_sorted_variables(self):
        """ make sure m.sorted_variables returns self.m.p """
        self.assertTrue(self.m.sorted_variables[0] is self.m.p)

    def test_parameter_tensors(self):
        """ make sure m.parameters returns a list of _tensor """
        self.assertTrue(self.m.parameter_tensors[0] is self.m.p.tensor)

    def testAssign(self):
        self.m.initialize()
        before = self.m.p.value
        self.m.p = 3.0
        self.assertTrue(isinstance(self.m.p, hb.param.Variable))
        self.m.initialize()
        after = self.m.p.value
        self.assertFalse(np.allclose(before, np.array([3.0])))
        self.assertTrue(np.allclose(after, np.array([3.0])))
        ##self.assertTrue(self.m.get_tensor_dict()[self.m.p._tf_array] == 2.0)

    def testReplacement(self):
        old_p = self.m.p
        new_p = hb.param.Variable(3)
        self.m.p = new_p
        self.assertFalse(old_p.highest_parent is self.m)

    def testHighestParent(self):
        self.assertTrue(self.m.p.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.p.name == 'p')

    def testTFMode(self):
        self.assertTrue(isinstance(self.m.p, hb.param.Variable))
        with self.m.tf_mode():
            self.assertTrue(isinstance(self.m.p, tf.Variable))
            self.assertFalse(isinstance(self.m.p, tf.Tensor))

    def test_local_variables(self):
        local_vars = self.m.local_variables
        self.assertTrue(self.m.q in local_vars)
        self.assertTrue(self.m.r in local_vars)

    def test_feed_size(self):
        feed_size = self.m.feed_size
        feed_size_manual = self.m.q.shape[0] + self.m.r.shape[0]*self.m.r.shape[1]
        self.assertTrue(feed_size == feed_size_manual)

    def test_feed_variable(self):
        """ make sure Variable.feed works """
        val = np.ones(1)
        tensor = tf.Variable(val, dtype=float_type)
        self.m.q.feed(tensor)
        self.m._session.run(tf.initialize_variables([tensor]))
        self.assertTrue(np.allclose(val, self.m.q.value))

    def test_feed_parameterized(self):
        """ make sure Parameterized.feed works """
        val = self.rng.randn(self.m.feed_size,10)
        tensor = tf.Variable(val, dtype=float_type)
        self.m.feed(tensor)
        self.m._session.run(tf.initialize_variables([tensor]))
        self.assertTrue(np.allclose(val[:3], self.m.q.value))
        self.assertTrue(np.allclose(val[3:], self.m.r.value))

class ParamTestLayered(unittest.TestCase):
    """ test Layered LocalVariable """
    def setUp(self):
        self.rng = np.random.RandomState(0)
        tf.reset_default_graph()
        self.m = hb.model.Model()
        self.m.p = hb.param.Variable([3],   n_layers = [2,3])
        self.m.q = hb.param.Variable([3],   n_layers = [2,3], collections=graph_key.LOCAL)
        self.m.r = hb.param.Variable([5,4], n_layers = [2,3], collections=graph_key.LOCAL)

    def test_global(self):
        """ make sure Parameterized.feed works """
        val = self.rng.randn(2,3,3)
        self.m.p = val
        self.m.initialize()
        self.assertTrue(np.allclose(val, self.m.p.value))

    def test_feed(self):
        """ make sure Parameterized.feed works """
        val = self.rng.randn(2,3,self.m.feed_size,10)
        tensor = tf.Variable(val, dtype=float_type)
        self.m.feed(tensor)
        self.m._session.run(tf.initialize_variables([tensor]))
        self.assertTrue(np.allclose(val[:,:,:3], self.m.q.value))
        self.assertTrue(np.allclose(val[:,:,3:], self.m.r.value))


class ParamTestsDeeper(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        tf.reset_default_graph()
        self.m = hb.model.Model()
        self.m.foo = hb.param.Parameterized()
        self.m.foo.bar = hb.param.Parameterized()
        self.m.foo.bar.baz = hb.param.Variable(1)
        self.m.foo.bar.q = hb.param.Variable([3], collections=graph_key.LOCAL)
        self.m.foo.bar.r = hb.param.Variable([5,4], collections=graph_key.LOCAL)

    def testHighestParent(self):
        self.assertTrue(self.m.foo.highest_parent is self.m)
        self.assertTrue(self.m.foo.bar.highest_parent is self.m)
        self.assertTrue(self.m.foo.bar.baz.highest_parent is self.m)

    def testReplacement(self):
        old_p = self.m.foo.bar.baz
        new_p = hb.param.Variable(3)
        self.m.foo.bar.baz = new_p
        self.assertFalse(old_p.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.foo.bar.name == 'bar')
        self.assertTrue(self.m.foo.bar.baz.name == 'baz')

    def testAssign(self):
        self.m.initialize()
        before = self.m.foo.bar.baz.value
        self.m.foo.bar.baz = 3.0
        self.assertTrue(isinstance(self.m.foo.bar.baz, hb.param.Variable))
        self.m.initialize()
        after = self.m.foo.bar.baz.value
        self.assertFalse(np.allclose(before, np.array([3.0])))
        self.assertTrue(np.allclose(after, np.array([3.0])))
        ##self.assertTrue(self.m.get_tensor_dict()[self.m.p._tf_array] == 2.0)

    def test_feed_parameterized(self):
        """ make sure Parameterized.feed works """
        val = self.rng.randn(self.m.feed_size,10)
        tensor = tf.Variable(val, dtype=float_type)
        self.m.feed(tensor)
        self.m._session.run(tf.initialize_variables([tensor]))
        self.assertTrue(np.allclose(val[:3], self.m.foo.bar.q.value))
        self.assertTrue(np.allclose(val[3:], self.m.foo.bar.r.value))


class TestParamList(unittest.TestCase):
    def test_construction(self):
        hb.param.ParamList([])
        hb.param.ParamList([hb.param.Variable(1)])
        with self.assertRaises(AssertionError):
            hb.param.ParamList([hb.param.Variable(1), 'stringsnotallowed'])

    def test_naming(self):
        p1 = hb.param.Variable([2])
        p2 = hb.param.Variable([3, 5])
        hb.param.ParamList([p1, p2])
        self.assertTrue(p1.name == 'item0')
        self.assertTrue(p2.name == 'item1')

    def test_connected(self):
        p1 = hb.param.Variable([2])
        p2 = hb.param.Variable([3, 5])
        l = hb.param.ParamList([p1, p2])
        x = l.sorted_variables
        self.assertTrue(p1 in x)
        self.assertTrue(p2 in x)

    def test_setitem(self):
        p1 = hb.param.Variable([1])
        p2 = hb.param.Variable([2,3])
        m = hb.model.Model()
        m.l = hb.param.ParamList([p1, p2])

        m.l[0] = 1.0
        m.initialize()
        self.assertTrue(np.allclose(p1.value, 1.0))

        with self.assertRaises(TypeError):
            m.l[0] = hb.param.Variable(12)

    def test_append(self):
        p1 = hb.param.Variable(1)
        p2 = hb.param.Variable([2,3])
        l = hb.param.ParamList([p1])
        l.append(p2)
        self.assertTrue(p2 in l.sorted_variables)

        with self.assertRaises(AssertionError):
            l.append('foo')

    def test_with_parameterized(self):
        pzd = hb.param.Parameterized()
        p = hb.param.Variable([1])
        pzd.p = p
        m = hb.model.Model()
        m.l = hb.param.ParamList([pzd])

        # test assignment:
        m.l[0].p = 5
        m.initialize()
        self.assertTrue(np.allclose(p.value, 5.0))

        # test to make sure tf_mode get turned on and off
        self.assertFalse(pzd._tf_mode)
        with m.tf_mode():
            self.assertTrue(pzd._tf_mode)
        self.assertFalse(pzd._tf_mode)

    # TODO prepare model.optimize
    """
    def test_in_model(self):
        class Foo(GPflow.model.Model):
            def __init__(self):
                Henbun.model.Model.__init__(self)
                self.l = Henbun.param.ParamList([
                    Henbun.param.Variable(1), Henbun.param.Variable(12)])

            def build_likelihood(self):
                return -reduce(tf.add, [tf.square(x) for x in self.l])

        m = Foo()
        self.assertTrue(m.get_tensor_dict().size == 2)
        m.optimize(disp=False)
        atol = 1e-6 if np_float_type is np.float32 else 1e-8
        self.assertTrue(np.allclose(m.get_tensor_dict(), 0., atol=atol))

    """

'''
class TestPickleAndDict(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        self.m = Henbun.model.Model()
        self.m.x = Henbun.param.Variable(X)
        self.m.y = Henbun.param.Variable(Y, 'local_param')

    def test_param(self):
        # pickle only the param
        s1 = pickle.dumps(self.m.x)
        m1 = pickle.loads(s1)
        self.assertTrue(np.allclose(self.m.x._array, m1._array))

    def test_model(self):
        # pickle and reload the model
        s1 = pickle.dumps(self.m)
        m1 = pickle.loads(s1)

        d1 = self.m.get_variable_dict()
        d2 = m1.get_variable_dict()
        for key, val in d1.items():
            assert np.all(val == d2[key])

class TestDictEmpty(unittest.TestCase):
    def setUp(self):
        self.m = Henbun.model.Model()

    def test(self):
        d = self.m.get_variable_dict()
        self.assertTrue(len(d.keys()) == 0)
        self.m.set_variable_dict(d)


class TestDictSimple(unittest.TestCase):
    def setUp(self):
        self.m = Henbun.model.Model()
        self.m.p1 = Henbun.param.Variable(np.random.randn(3, 2))
        self.m.p2 = Henbun.param.Variable(np.random.randn(10))

    def test(self):
        d = self.m.get_variable_dict()
        self.assertTrue(len(d.keys()) == 2)
        state1 = self.m.get_variable_dict().copy()
        self.m.set_variable_dict(d)
        state2 = self.m.get_variable_dict()
        for key in state2:
            self.assertTrue(np.all(state1[key] == state2[key]))

'''
if __name__ == '__main__':
    unittest.main()
