import unittest
import numpy as np
import MAS_library as MASL

FLOAT = MASL.FLOAT_type()

class TestMASL(unittest.TestCase):

    ######## cython MAS ########
    def test_NGP(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        MASL.MA(pos, delta, BoxSize, 'NGP', W=None)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=20)

    def test_NGPW(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(np.float32)
        delta = np.zeros((dims,dims,dims), dtype=np.float32)
        W = np.ones(particles, dtype=np.float32)*3.0
        MASL.MA(pos, delta, BoxSize, 'NGP', W=W)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=20)

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


    ######## MAS_c #########
    def test_NGPc3D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(FLOAT)
        delta = np.zeros((dims,dims,dims), dtype=FLOAT)
        MASL.NGPc3D(pos, delta, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=20)

    def test_NGPWc3D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(FLOAT)
        delta = np.zeros((dims,dims,dims), dtype=FLOAT)
        W = np.ones(particles, dtype=FLOAT)*3.0
        MASL.NGPWc3D(pos, delta, W, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=20)

    def test_NGPc2D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,2)).astype(FLOAT)
        delta = np.zeros((dims,dims), dtype=FLOAT)
        MASL.NGPc2D(pos, delta, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=20)

    def test_NGPWc2D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,2)).astype(FLOAT)
        delta = np.zeros((dims,dims), dtype=FLOAT)
        W = np.ones(particles, dtype=FLOAT)*3.0
        MASL.NGPWc2D(pos, delta, W, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=20)

    def test_CICc3D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(FLOAT)
        delta = np.zeros((dims,dims,dims), dtype=FLOAT)
        MASL.CICc3D(pos, delta, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=8)

    def test_CICWc3D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(FLOAT)
        delta = np.zeros((dims,dims,dims), dtype=FLOAT)
        W = np.ones(particles, dtype=FLOAT)*3.0
        MASL.CICWc3D(pos, delta, W, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=8)

    def test_CICc2D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,2)).astype(FLOAT)
        delta = np.zeros((dims,dims), dtype=FLOAT)
        MASL.CICc2D(pos, delta, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=8)

    def test_CICWc2D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,2)).astype(FLOAT)
        delta = np.zeros((dims,dims), dtype=FLOAT)
        W = np.ones(particles, dtype=FLOAT)*3.0
        MASL.CICWc2D(pos, delta, W, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=8)

    def test_TSCc3D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(FLOAT)
        delta = np.zeros((dims,dims,dims), dtype=FLOAT)
        MASL.TSCc3D(pos, delta, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=8)

    def test_TSCWc3D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,3)).astype(FLOAT)
        delta = np.zeros((dims,dims,dims), dtype=FLOAT)
        W = np.ones(particles, dtype=FLOAT)*3.0
        MASL.TSCWc3D(pos, delta, W, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=8)

    def test_TSCc2D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,2)).astype(FLOAT)
        delta = np.zeros((dims,dims), dtype=FLOAT)
        MASL.TSCc2D(pos, delta, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/particles, 1.0, places=8)

    def test_TSCWc2D(self):
        particles, BoxSize, dims, seed = 1000, 1.0, 64, 1
        np.random.seed(seed)
        pos = np.random.random((particles,2)).astype(FLOAT)
        delta = np.zeros((dims,dims), dtype=FLOAT)
        W = np.ones(particles, dtype=FLOAT)*3.0
        MASL.TSCWc2D(pos, delta, W, BoxSize, 2)
        suma = np.sum(delta, dtype=np.float64)
        self.assertAlmostEqual(suma/(3.0*particles), 1.0, places=8)
        
if __name__== '__main__':
    unittest.main()
