#!/bin/bash
# Trains DBN on MNIST.
train_deepnet='python ../../trainer.py'
echo "RBM 1"
${train_deepnet} mnist_rbm1.pbtxt train.pbtxt eval.pbtxt || exit 1
echo "RBM 2"
${train_deepnet} mnist_rbm2.pbtxt train.pbtxt eval.pbtxt || exit 1
echo "RBM 3"
${train_deepnet} mnist_rbm3.pbtxt train.pbtxt eval.pbtxt || exit 1
echo "Classifier"
${train_deepnet} mnist_classifier.pbtxt train_classifier.pbtxt eval.pbtxt

