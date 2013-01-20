#!/bin/bash
# Trains DBN on MNIST.

echo "Autoencoder 1"
train_deepnet model_layer1.pbtxt train.pbtxt eval.pbtxt
echo "Autoencoder 2"
train_deepnet model_layer2.pbtxt train.pbtxt eval.pbtxt
echo "Classifier"
train_deepnet classifier.pbtxt train.pbtxt eval.pbtxt

