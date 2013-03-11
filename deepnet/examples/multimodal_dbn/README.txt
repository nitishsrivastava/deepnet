############################
MULTIMODAL DEEP BELIEF NETS

Nitish Srivastava
University of Toronto
############################
This code trains a Multimodal DBN on the MIR-Flickr dataset.
The implementation uses GPUs to accelerate training.

(1) GET DATA
  - Download preprocessed data into a place with lots of disk space.
  $ cd path/to/data
  $ wget http://www.cs.toronto.edu/~nitish/multimodal/flickr_data.tar.gz
  $ tar -xvzf flickr_data.tar.gz 

(2) TRAIN MULTIMODAL DBN
  - Change to the directory containing this file.
  - Edit paths in runall_dbn.sh
  - train dbn
  $ ./runall_dbn.sh

This implementation has been tested on Ubuntu 12.04 using CUDA 4.2.

Last edited -
Sun Mar 10 01:26:01 EST 2013
