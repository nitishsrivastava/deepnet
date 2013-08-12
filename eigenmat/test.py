import unittest
import eigenmat as mat
import numpy as np

class TestEigenMat(unittest.TestCase):

  def setUp(self):
    mat.EigenMatrix.init_random(seed=1)

  def test_add(self):
    x = np.random.randn(10, 10)
    y = np.random.randn(10, 10)
    eig_x = mat.EigenMatrix(x)
    eig_y = mat.EigenMatrix(y)
    eig_z = mat.empty(x.shape)

    z = x + y  # Numpy add.
    eig_x.add(eig_y, target=eig_z)  # EigenMat add.
    
    diff = ((eig_z.asarray() - z)**2).sum()
    self.assertAlmostEqual(diff, 0)

  def test_dot(self):
    x = np.random.randn(500, 1000)
    y = np.random.randn(1000, 600)
    eig_x = mat.EigenMatrix(x)
    eig_y = mat.EigenMatrix(y)
    eig_z = mat.empty((x.shape[0], y.shape[1]))

    z = x.dot(y)
    mat.dot(eig_x, eig_y, target=eig_z)

    diff = ((eig_z.asarray() - z)**2).sum()
    self.assertAlmostEqual(diff, 0, places=5)

  def test_dot_transposed(self):
    x = np.random.randn(500, 1000)
    y = np.random.randn(600, 1000)
    eig_x = mat.EigenMatrix(x)
    eig_y = mat.EigenMatrix(y)
    eig_z = mat.empty((x.shape[0], y.shape[0]))

    z = x.dot(y.T)
    mat.dot(eig_x, eig_y.T, target=eig_z)

    diff = ((eig_z.asarray() - z)**2).sum()
    self.assertAlmostEqual(diff, 0, places=5)

  def test_sum_by_axis(self):
    x = 1.1 + np.random.randn(10, 1000)
    y = np.zeros((1, 1000))
    z = np.zeros((10, 1))
    eig_x = mat.EigenMatrix(x)
    eig_y = mat.EigenMatrix(y)
    eig_z = mat.EigenMatrix(z)

    eig_x.sum(axis=0, target=eig_y)
    eig_x.sum(axis=1, target=eig_z)
    diff = ((eig_y.asarray() - x.sum(axis=0).reshape(1, -1))**2).sum()
    self.assertAlmostEqual(diff, 0, places=5)
    diff = ((eig_z.asarray() - x.sum(axis=1).reshape(-1, 1))**2).sum()
    self.assertAlmostEqual(diff, 0, places=5)


  def test_apply_softmax(self):
    x = np.random.randn(100, 10)
    eig_x = mat.EigenMatrix(x)
    eig_y = mat.empty((100, 10))

    eig_x.apply_softmax(target=eig_y)
    
    y = np.exp(x - x.max(axis=0))
    y /= y.sum(axis=0)

    diff = ((eig_y.asarray() - y)**2).sum()
    self.assertAlmostEqual(diff, 0, places=5)

if __name__ == '__main__':
  unittest.main()



