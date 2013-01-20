# Extract representation from different layers and write them to disk.
#!/bin/bash
# 
# python ../../extract_neural_net_representation.py <model_file> <train_op> <output_dir> <list of layer names>
python ../../extract_neural_net_representation.py\
  /ais/gobi3/u/nitish/mnist/models/mnist_3layer_relu_LAST \
  train.pbtxt \
  /ais/gobi3/u/nitish/mnist/reps/3layer_relu \
  hidden1 hidden2 hidden3 output_layer
