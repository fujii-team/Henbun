import tensorflow as tf
import numpy as np
import unittest
import Henbun as hb
from Henbun._settings import settings
from Henbun.param import graph_key
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class TestData(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.x = self.rng.randn(3,2)
        self.y = self.rng.randn(4,3)
        # --- define model ---
        tf.reset_default_graph()
        self.m = hb.model.Model()
        self.m.p = hb.param.Parameterized()
        self.m.x = hb.param.Data(self.x)
        self.m.p.y = hb.param.Data(self.y)

    def test_value(self):
        x = self.m.x.value
        self.assertTrue(np.allclose(x, self.x))
        y = self.m.p.y.value
        self.assertTrue(np.allclose(y, self.y))

    def test_replacement(self):
        x = self.rng.randn(3,2)
        self.m.x = x
        self.assertTrue(np.allclose(x, self.m.x.value))
        y = self.rng.randn(4,3)
        self.m.p.y = y
        self.assertTrue(np.allclose(y, self.m.p.y.value))
        # assign different size of data
        with self.assertRaises(ValueError):
            self.m.p.y = self.rng.randn(4,4)


if __name__ == "__main__":
    unittest.main()
