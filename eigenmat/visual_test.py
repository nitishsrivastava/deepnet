import matplotlib.pyplot as plt
import numpy as np
plt.ion()
import eigenmat as mat
mat.EigenMatrix.init_random(seed=1)
plt.figure(1)
plt.clf()
x = mat.empty((100, 100))
x.fill_with_randn()
plt.hist(x.asarray().flatten(), 100)

plt.figure(2)
plt.clf()
y = np.random.randn(100, 100)
plt.hist(y.flatten(), 100)

raw_input('Press Enter.')
