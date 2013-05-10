from layer import *
from logistic_layer import *
from linear_layer import *
from softmax_layer import *

class FastDropoutLayer(Layer):
  pass
class FastDropoutLogisticLayer(FastDropoutLayer, LogisticLayer):
  pass
class FastDropoutLinearLayer(FastDropoutLayer, LinearLayer):
  pass
class FastDropoutSoftmaxLayer(FastDropoutLayer, LinearLayer):
  pass
