#!/bin/bash
# Trains DBN on MNIST.
train_deepnet='python ../../trainer.py'
echo "Autoencoder 1"
${train_deepnet} model_layer1.pbtxt train.pbtxt eval.pbtxt || exit 1
echo "Autoencoder 2"
${train_deepnet} model_layer2.pbtxt train.pbtxt eval.pbtxt || exit 1
echo "Classifier"
${train_deepnet} classifier.pbtxt train.pbtxt eval.pbtxt

