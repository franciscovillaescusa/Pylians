import unittest
import MAS_library as MASL
import numpy as np


class TestMASL(unittest.TestCase):
    
    def test_NGP(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        MASL.MA(pos, delta, BoxSize, 'NGP', W=None)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=8)

    def test_NGPW(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        W = np.ones(particles, dtype=np.float32)*3.0
        MASL.MA(pos, delta, BoxSize, 'NGP', W=W)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=8)

    def test_CIC(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        MASL.MA(pos, delta, BoxSize, 'CIC', W=None)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=8)

    def test_CICW(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        W = np.ones(particles, dtype=np.float32)*3.0
        MASL.MA(pos, delta, BoxSize, 'CIC', W=W)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=8)

    def test_TSC(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        MASL.MA(pos, delta, BoxSize, 'TSC', W=None)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=8)

    def test_TSCW(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        W = np.ones(particles, dtype=np.float32)*3.0
        MASL.MA(pos, delta, BoxSize, 'TSC', W=W)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=8)

    def test_PCS(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        MASL.MA(pos, delta, BoxSize, 'PCS', W=None)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=8)

    def test_PCSW(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        W = np.ones(particles, dtype=np.float32)*3.0
        MASL.MA(pos, delta, BoxSize, 'PCS', W=W)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=8)



if __name__== '__main__':
    unittest.main()
