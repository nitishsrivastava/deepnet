import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
plt.ion()

fig_id = 0
def GetFigId():
  globals()['fig_id'] += 1
  return globals()['fig_id'] - 1

def show_model_state(model, step):
  for i, node in enumerate(model.node_list):
    dims = int(np.floor(np.sqrt(node.state.shape[0])))
    display_w(node.sample.asarray(), dims, 10, 10, i, title=node.name)


def show(mat, fig=1, title=''):
  plt.figure(fig)
  plt.clf()
  plt.imshow(mat, interpolation='nearest')
  plt.suptitle(title)
  plt.colorbar()
  plt.draw()

def scatter(Y, s=20, c='b', fig=1):
  plt.figure(fig)
  plt.clf()
  plt.scatter(Y[:,0], Y[:,1], s, c)
  plt.draw()

def show_hist(mat, fig):
  plt.figure(fig)
  plt.clf()
  plt.hist(mat.flatten(), 100)
  plt.draw()

def show_stats(edge, fig, title):
  plt.figure(fig)
  plt.clf()
  plt.suptitle(title)
  plt.hist(edge.params['weight'].asarray().flatten(), 100)
  plt.draw()

def display_hidden(state, fig, title, log=False, prob=True):
  plt.figure(fig)
  plt.clf()
  plt.suptitle(title)
  plt.subplot(1, 3, 1)
  plt.hist(state.mean(axis=1), 100)
  if prob:
    plt.xlim([0, 1])
  plt.title('Mean Activation')
  plt.subplot(1, 3, 2)
  plt.hist(state.flatten(), 100, log=log)
  if prob:
    plt.xlim([-0.1, 1.1])
  plt.title('Activation')
  plt.subplot(1, 3, 3)
  plt.imshow(state, cmap = plt.cm.gray, interpolation='nearest', vmax=1, vmin=0)
  plt.title('State')
  plt.draw()

def display_wsorted(w, s, r, c, fig, vmax=None, vmin=None, dataset='mnist',
                    title='weights_sorted'):

  if dataset == 'norb':
    numvis = 4096
  else:
    numvis = w.shape[0]
  numhid = w.shape[1]
  sc = s
  sr = numvis/s
  padding = numhid - r*c
  if isinstance(w, np.ndarray):
    w = w.T[:, :sr*sc]
  else:
    w = w.asarray().T[:, :sr*sc]

  vh = w.reshape(sr*numhid, sc)
  pvh = np.zeros((sr, sc, r, c))
  pvh2 = np.zeros((sr*r, sc*c))
  norm_list = []
  for i in range(r):
    for j in range(c):
      pvh[:,:, i, j] = vh[ (i*c+j)*sr : (i*c+j+1)*sr ,:]
      norm = (pvh[:,:,i,j]**2).sum()
      norm_list.append((norm, i, j))
  norm_list.sort(reverse = True)
  index = 0
  for norm, i, j in norm_list:
    ii = index/c
    jj = index%c
    pvh2[ii*sr:(ii+1)*sr , jj*sc:(jj+1)*sc] = pvh[:,:,i,j]
    index+=1
  plt.figure(fig)
  plt.clf()

  plt.suptitle(title)
  # vmax = 0.5
  # vmin = -0.5
  plt.imshow(pvh2, cmap = plt.cm.gray, interpolation = 'nearest', vmax=vmax, vmin=vmin)
  scale = 1
  xmax = sc*c
  ymax = sr*r
  color = 'k'
  for x in range(0,c):
    plt.axvline(x=x*sc/scale,ymin=0,ymax=ymax/scale, color = color)
  for y in range(0,r):
    plt.axhline(y=y*sr/scale, xmin=0,xmax=xmax/scale, color = color)
  plt.draw()

  return pvh

def display_w(w, s, r, c, fig, vmax=None, vmin=None, dataset='mnist', title='weights'):

  if dataset == 'norb':
    numvis = 4096
  else:
    numvis = w.shape[0]
  numhid = w.shape[1]
  sc = s
  sr = numvis/s
  if isinstance(w, np.ndarray):
    vh = w.T[:,:sr*sc].reshape(sr*numhid, sc)
  else:
    vh = w.asarray().T[:,:sr*sc].reshape(sr*numhid, sc)
  pvh = np.zeros((sr*r, sc*c))
  for i in range(r):
    for j in range(c):
      pvh[i*sr:(i+1)*sr , j*sc:(j+1)*sc] = vh[ (i*c+j)*sr : (i*c+j+1)*sr ,:]
  plt.figure(fig)
  plt.clf()
  plt.title(title)
  plt.imshow(pvh, cmap = plt.cm.gray, interpolation = 'nearest', vmax=vmax, vmin=vmin)
  scale = 1
  xmax = sc*c
  ymax = sr*r
  color = 'k'
  if r > 1:
    for x in range(0,c):
      plt.axvline(x=x*sc/scale, ymin=0,ymax=ymax/scale, color = color)
  if c > 1:
    for y in range(0,r):
      plt.axhline(y=y*sr/scale, xmin=0,xmax=xmax/scale, color = color)
  plt.draw()

  return pvh

def display_convw2(w, s, r, c, fig, title='conv_filters'):
  """w: num_filters X sizeX**2 * num_colors."""
  num_f, num_d = w.shape
  assert s**2 * 3 == num_d
  pvh = np.zeros((s*r, s*c, 3))
  for i in range(r):
    for j in range(c):
      pvh[i*s:(i+1)*s, j*s:(j+1)*s, :] = w[i*c + j, :].reshape(3, s, s).T
  mx = pvh.max()
  mn = pvh.min()
  pvh = 255*(pvh - mn) / (mx-mn)
  pvh = pvh.astype('uint8')
  plt.figure(fig)
  plt.suptitle(title)
  plt.imshow(pvh, interpolation="nearest")
  scale = 1
  xmax = s * c
  ymax = s * r
  color = 'k'
  for x in range(0, c):
    plt.axvline(x=x*s/scale, ymin=0, ymax=ymax/scale, color=color)
  for y in range(0, r):
    plt.axhline(y=y*s/scale, xmin=0, xmax=xmax/scale, color=color)
  plt.draw()
  return pvh

def display_convw(w, s, r, c, fig, vmax=None, vmin=None, dataset='mnist', title='conv_filters'):

  """
  w2 = np.zeros(w.shape)
  d = w.shape[1]/3
  print w.shape
  for i in range(w.shape[0]):
    for j in range(w.shape[1]/3):
      w2[i, j] = w[i, 3*j]
      w2[i, j + d] = w[i, 3*j+1]
      w2[i, j + 2*d] = w[i, 3*j+2]
  w = w2
  """

  numhid = w.shape[0]
  size_x = s
  size_y = s    # For now.
  num_channels = w.shape[1] / (size_x*size_y)
  assert num_channels == 3
  assert w.shape[1] % size_x*size_y == 0
  if isinstance(w, np.ndarray):
    vh = w.reshape(size_x*numhid*num_channels, size_y)
  else:
    vh = w.asarray().reshape(size_x*numhid*num_channels, size_y)
  pvh = np.zeros((size_x*r, size_y*c, num_channels))
  for i in range(r):
    for j in range(c):
      for ch in range(num_channels):
        pvh[i*size_x:(i+1)*size_x, j*size_y:(j+1)*size_y, ch] = \
            vh[(num_channels*(i*c+j)+ch)*size_x:(num_channels*(i*c+j)+ch+1)*size_x,:]

  # pvh /= np.std(pvh)
  plt.figure(fig)
  plt.clf()
  plt.title(title)
  plt.imshow(pvh, vmax=vmax, vmin=vmin)
  scale = 1
  xmax = size_x*c
  ymax = size_y*r
  color = 'k'
  for x in range(0, c):
    plt.axvline(x=x*size_x/scale, ymin=0,ymax=ymax/scale, color = color)
  for y in range(0, r):
    plt.axhline(y=y*size_y/scale, xmin=0,xmax=xmax/scale, color = color)
  plt.draw()

  return pvh
