import unittest
import integration_library as IL
import numpy as np


class TestIntegration(unittest.TestCase):

    def test_RK4_1(self):

        # This routine computes \int_0^5 (x^2 + 2*x + 3)dx = 81.66666666666
        yinit = np.array([0.0], dtype=np.float64)
        x1, x2 = 0.0, 5.0
        nstep = 10000
        yout = IL.RK4_example(yinit, x1, x2, nstep)[0]
        self.assertAlmostEqual(yout, 81.66666666666666, 10)

    def test_RK4_2(self):

        # This routine computes \int_0^5 (sin(x)+exp(-x^2))dx = 1.60256474
        # but having the function f(x) = sin(x)+exp(-x^2) evaluated only
        # in a few points. To compute f(x) in an arbitrary point, the input
        # points are interpolated
        yinit = np.array([0.0], dtype=np.float64)
        x1, x2 = 0.0, 5.0
        nstep = 100000
        x = np.linspace(x1, x2, 10000)
        y = np.sin(x) + np.exp(-x**2)
        yout = IL.RK4_example2(yinit, x1, x2, nstep, x, y)[0]
        self.assertAlmostEqual(yout, 1.6025647399881, 4)

    def test_odeint_1(self):

        # This routine computes \int_0^5 (x^2 + 2*x + 3)dx = 81.66666666666
        yinit = np.array([0.0], dtype=np.float64)
        x1, x2 = 0.0, 5.0
        eps = 1e-15
        h1 = 1e-10
        hmin = 0.0
        yout = IL.odeint_example1(yinit, x1, x2, eps, h1, hmin,
                                  verbose=False)[0]
        self.assertAlmostEqual(yout, 81.66666666666666, 12)
        

    def test_odeint_2(self):

        # This routine computes \int_0^5 (sin(x)+exp(-x^2))dx = 1.60256474
        # but having the function f(x) = sin(x)+exp(-x^2) evaluated only
        # in a few points. To compute f(x) in an arbitrary point, the input
        # points are interpolated
        yinit = np.array([0.0], dtype=np.float64)
        x1, x2 = 0.0, 5.0
        eps = 1e-15
        h1 = 1e-10
        hmin = 0.0
        x = np.linspace(x1, x2, 10000)
        y = np.sin(x) + np.exp(-x**2)
        yout = IL.odeint_example2(yinit, x1, x2, eps, h1, hmin, x, y,
                                  verbose=False)[0]
        self.assertAlmostEqual(yout, 1.6025647399881, 7)

if __name__=='__main__':
    unittest.main()
