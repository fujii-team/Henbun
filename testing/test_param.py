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
        self.m.p = hb.param.Variable([1], transform=hb.transforms.positive)
        # local
        self.m.q = hb.param.Variable([3], collections=graph_key.LOCAL)
        # layered local
        self.m.r = hb.param.Variable([5,4], collections=graph_key.LOCAL)
        # batched_global
        self.m.s = hb.param.Variable([5,4], n_batch=2)

    def test_sorted_variables(self):
        """ make sure m.sorted_variables returns self.m.p """
        self.assertTrue(self.m.sorted_variables[0] is self.m.p)
        self.assertTrue(len(self.m.sorted_variables) == 4)

    def test_get_variables(self):
        """ make sure m.parameters returns a list of _tensor """
        self.assertTrue(self.m.p._tensor in self.m.get_tf_variables())
        self.assertTrue(self.m.q._tensor not in self.m.get_tf_variables(hb.param.graph_key.VARIABLES))

    def testAssign(self):
        before = self.m.p.value
        self.m.p = 3.0
        self.assertTrue(self.m.p._assigned)
        self.assertTrue(isinstance(self.m.p, hb.param.Variable))
        after = self.m.p.value
        self.assertFalse(self.m.p._assigned)
        self.assertFalse(np.allclose(before, np.array([3.0])))
        self.assertTrue(np.allclose(after, np.array([3.0])))
        ##self.assertTrue(self.m.get_tensor_dict()[self.m.p._tf_array] == 2.0)

    def testValue(self):
        """ Make sure initialize certainly works if self.m.p.value is called """
        # make sure initialize op certainly assigned
        self.assertTrue(self.m.p._assigned)
        # call self.m.p.value
        a = self.m.p.value
        # make sure assigned flag removed.
        self.assertFalse(self.m.p._assigned)


    def testReplacement(self):
        old_p = self.m.p
        new_p = hb.param.Variable(3)
        self.m.p = new_p
        self.assertTrue(new_p.highest_parent is self.m)
        self.assertFalse(old_p.highest_parent is self.m)

    def testHighestParent(self):
        self.assertTrue(self.m.p.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.p.name == 'p')

    def testTFMode(self):
        self.assertTrue(isinstance(self.m.p, hb.param.Variable))
        with self.m.tf_mode():
            self.assertTrue(isinstance(self.m.p, tf.Tensor))

    def test_local_variables(self):
        local_vars = self.m.get_variables(graph_key.LOCAL)
        self.assertTrue(self.m.q in local_vars)
        self.assertTrue(self.m.r in local_vars)

    def test_feed_size(self):
        feed_size = self.m.feed_size
        feed_size_manual = self.m.q.shape[0] + self.m.r.shape[0]*self.m.r.shape[1]
        self.assertTrue(feed_size == feed_size_manual)

    def test_feed_variable(self):
        """ make sure Variable.feed works """
        val_q = np.ones((10,3))
        tensor_q = tf.Variable(val_q, dtype=float_type)
        val_r = np.ones((10,5*4))
        tensor_r = tf.Variable(val_r, dtype=float_type)
        val_s = np.ones((2,5,4))
        self.m.q.feed(tensor_q)
        self.m.r.feed(tensor_r)
        self.m.s = val_s
        self.m._session.run(tf.variables_initializer([tensor_q, tensor_r]))
        self.assertTrue(np.allclose(val_q, self.m.q.value))
        self.assertTrue(np.allclose(val_r.flatten(), self.m.r.value.flatten()))
        self.assertTrue(np.allclose(val_s, self.m.s.value))

    def test_feed_parameterized(self):
        """ make sure Parameterized.feed works """
        val = self.rng.randn(10, self.m.feed_size)
        tensor = tf.Variable(val, dtype=float_type)
        self.m.feed(tensor)
        self.m._session.run(tf.variables_initializer_variables([tensor]))
        self.assertTrue(np.allclose(val[:,:3], self.m.q.value))
        self.assertTrue(np.allclose(val[:,3:].flatten(), self.m.r.value.flatten()))

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
        self.assertTrue(np.allclose(val, self.m.p.value))

    def test_feed(self):
        """ make sure Parameterized.feed works """
        val = self.rng.randn(2,3,10,self.m.feed_size)
        tensor = tf.Variable(val, dtype=float_type)
        self.m.feed(tensor)
        self.m._session.run(tf.variables_initializer([tensor]))
        self.assertTrue(np.allclose(val[:,:,:,:3].flatten(), self.m.q.value.flatten()))
        self.assertTrue(np.allclose(val[:,:,:,3:].flatten(), self.m.r.value.flatten()))


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

    def testReplacement2(self):
        old_p = self.m.foo
        new_p = hb.param.Variable(3)
        self.m.foo = new_p
        self.assertTrue(new_p.highest_parent is self.m)
        self.assertFalse(old_p.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.foo.bar.name == 'bar')
        self.assertTrue(self.m.foo.bar.baz.name == 'baz')

    def testAssign(self):
        before = self.m.foo.bar.baz.value
        self.m.foo.bar.baz = 3.0
        self.assertTrue(isinstance(self.m.foo.bar.baz, hb.param.Variable))
        after = self.m.foo.bar.baz.value
        self.assertFalse(np.allclose(before, np.array([3.0])))
        self.assertTrue(np.allclose(after, np.array([3.0])))
        ##self.assertTrue(self.m.get_tensor_dict()[self.m.p._tf_array] == 2.0)

    def test_feed_parameterized(self):
        """ make sure Parameterized.feed works """
        val = self.rng.randn(10, self.m.feed_size)
        tensor = tf.Variable(val, dtype=float_type)
        self.m.feed(tensor)
        self.m._session.run(tf.variables_initializer([tensor]))
        self.assertTrue(np.allclose(val[:,:3].flatten(), self.m.foo.bar.q.value.flatten()))
        self.assertTrue(np.allclose(val[:,3:].flatten(), self.m.foo.bar.r.value.flatten()))


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

class InitTest(unittest.TestCase):
    def setUp(self):
        self.m = hb.model.Model()
        self.m.p = hb.param.Variable(shape=[10,20], mean=0.0, stddev=0.1)
        self.m.q = hb.param.Variable(shape=[10,20], mean=0.1, stddev=0.1)

    def test_init(self):
        self.assertTrue(np.all(self.m.p.value >-0.2))
        self.assertTrue(np.all(self.m.p.value < 0.2))
        self.assertTrue(np.all(self.m.q.value >-0.1))
        self.assertTrue(np.all(self.m.q.value < 0.3))

class DataTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        tf.reset_default_graph()
        self.m = hb.model.Model()
        self.m.foo = hb.param.Parameterized()
        self.m.foo.bar = hb.param.Parameterized()
        self.m.foo.bar.baz = hb.param.Variable(1)
        self.data = self.rng.randn(3,2)
        self.m.foo.bar.d1 = hb.param.Data(self.data)
        self.minibatch_data = self.rng.randn(10,3,2)
        self.m.foo.bar.d2 = hb.param.MinibatchData(self.minibatch_data)

    def test_get_feed_dict(self):
        index = self.rng.randint(0,10,3)
        feed_dict = self.m.get_feed_dict(index)
        self.assertTrue(np.allclose(feed_dict[self.m.foo.bar.d1._tensor],
                                            self.m.foo.bar.d1.data))
        self.assertTrue(np.allclose(feed_dict[self.m.foo.bar.d2._tensor],
                                            self.m.foo.bar.d2.data[index]))
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
