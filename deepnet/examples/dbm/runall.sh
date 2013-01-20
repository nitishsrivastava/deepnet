#!/bin/bash
# Trains a feed forward net on MNIST.
train_deepnet='python ../../trainer.py'
${train_deepnet} model.pbtxt train.pbtxt eval.pbtxt
