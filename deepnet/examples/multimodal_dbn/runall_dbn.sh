#!/bin/bash
# This script trains a Multimodal DBN using deepnet.
# Before running this script download and extract data from
# http://www.cs.toronto.edu/~nitish/multimodal

# Location of deepnet. EDIT this for your setup.
deepnet=$HOME/deepnet/deepnet

# Location of the downloaded data. This is also the place where learned models
# and representations extracted from them will be written. Should have lots of
# space ~30G. EDIT this for your setup.
prefix=/ais/gobi3/u/nitish/flickr

# Amount of gpu memory to be used for buffering data. Adjust this for your GPU.
# For a GPU with 6GB memory, this should be around 4GB.
# If you get 'out of memory' errors, try decreasing this.
gpu_mem=4G

# Amount of main memory to be used for buffering data. Adjust this according to
# your RAM. Having atleast 16G is ideal.
main_mem=20G

# Number of train/valid/test splits for doing classifiation experiments.
numsplits=5

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${prefix}/dbn_models
data_output_dir=${prefix}/dbn_reps
clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

# Set up paths, split data into with/without text.
echo Setting up data
python setup_data.py ${prefix} ${model_output_dir} ${data_output_dir} \
  ${gpu_mem} ${main_mem} ${numsplits} || exit 1

# Compute mean and variance of the data.
echo Computing mean / variance
if [ ! -e  ${prefix}/flickr_stats.npz ]
then
  python ${deepnet}/compute_data_stats.py ${prefix}/flickr.pbtxt \
    ${prefix}/flickr_stats.npz image_unlabelled || exit 1
fi
# IMAGE LAYER - 1.
#(
if ${clobber} || [ ! -e ${model_output_dir}/image_rbm1_LAST ]; then
  echo "Training first layer image RBM."
  python ${trainer} models/image_rbm1.pbtxt \
    trainers/dbn/train_CD_image_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/image_rbm1_LAST \
    trainers/dbn/train_CD_image_layer1.pbtxt image_hidden1 \
    ${data_output_dir}/image_rbm1_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# IMAGE LAYER - 2.
if ${clobber} || [ ! -e ${model_output_dir}/image_rbm2_LAST ]; then
  echo "Training second layer image RBM."
  python ${trainer} models/image_rbm2.pbtxt \
    trainers/dbn/train_CD_image_layer2.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/image_rbm2_LAST \
    trainers/dbn/train_CD_image_layer2.pbtxt image_hidden2 \
    ${data_output_dir}/image_rbm2_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# EXTRACT IMAGE REPRESENTATIONS CORRESPONDING TO DATA WITH NON-ZERO TEXT.
if ${clobber} || [ ! -e ${data_output_dir}/image_rbm1_LAST_nnz/data.pbtxt ]; then
  python ${extract_rep} ${model_output_dir}/image_rbm1_LAST \
    trainers/dbn/train_CD_image_layer1.pbtxt image_hidden1  \
    ${data_output_dir}/image_rbm1_LAST_nnz ${gpu_mem} ${main_mem} \
    ${prefix}/flickr_nnz.pbtxt || exit 1
fi
if ${clobber} || [ ! -e ${data_output_dir}/image_rbm2_LAST_nnz/data.pbtxt ]; then
  python ${extract_rep} ${model_output_dir}/image_rbm2_LAST \
    trainers/dbn/train_CD_image_layer2.pbtxt image_hidden2  \
    ${data_output_dir}/image_rbm2_LAST_nnz ${gpu_mem} ${main_mem} \
    ${data_output_dir}/image_rbm1_LAST_nnz/data.pbtxt || exit 1
fi
#)&
#(
# TEXT LAYER - 1.
if ${clobber} || [ ! -e ${model_output_dir}/text_rbm1_LAST ]; then
  echo "Training first layer text RBM."
  python ${trainer} models/text_rbm1.pbtxt \
    trainers/dbn/train_CD_text_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/text_rbm1_LAST \
    trainers/dbn/train_CD_text_layer1.pbtxt text_hidden1 \
    ${data_output_dir}/text_rbm1_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# TEXT LAYER - 2.
if ${clobber} || [ ! -e ${model_output_dir}/text_rbm2_LAST ]; then
  echo "Training second layer text RBM."
  python ${trainer} models/text_rbm2.pbtxt \
    trainers/dbn/train_CD_text_layer2.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/text_rbm2_LAST \
    trainers/dbn/train_CD_text_layer2.pbtxt text_hidden2 \
    ${data_output_dir}/text_rbm2_LAST ${gpu_mem} ${main_mem} || exit 1
fi
#)&
#wait;

# MERGE IMAGE AND TEXT DATA PBTXT FOR TRAINING JOINT RBM
if ${clobber} || [ ! -e ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt ]; then
  mkdir -p ${data_output_dir}/joint_rbm_LAST
  python merge_dataset_pb.py \
    ${data_output_dir}/image_rbm2_LAST_nnz/data.pbtxt \
    ${data_output_dir}/text_rbm2_LAST/data.pbtxt \
    ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt || exit 1
fi

# TRAIN JOINT RBM
if ${clobber} || [ ! -e ${model_output_dir}/joint_rbm_LAST ]; then
  echo "Training joint layer RBM."
  python ${trainer} models/joint_rbm.pbtxt \
    trainers/dbn/train_CD_joint_layer.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/joint_rbm_LAST \
    trainers/dbn/train_CD_joint_layer.pbtxt joint_hidden \
    ${data_output_dir}/joint_rbm_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# INFER TEXT PATHWAY CONDITIONED ON IMAGES.
if ${clobber} || [ ! -e ${data_output_dir}/generated_text/data.pbtxt ]; then
  echo "Inferring missing text"
  python sample_text.py models/multimodal_dbn.pbtxt \
    trainers/dbn/train_CD_image_layer1.pbtxt ${data_output_dir}/generated_text \
    ${prefix}/flickr_z.pbtxt ${gpu_mem} ${main_mem} || exit 1
fi

# COLLECT REPRESENTATIONS WITH/WITHOUT TEXT IN ONE DATA PBTXT.
if ${clobber} || [ ! -e ${data_output_dir}/dbn_all_layers/data.pbtxt ]; then
  echo "Collecting all representations"
  python collect_dbn_reps.py models/multimodal_dbn.pbtxt \
    ${data_output_dir}/dbn_all_layers \
    ${data_output_dir} \
    ${prefix} \
    ${gpu_mem} ${main_mem} || exit 1
fi

# SPLIT INTO TRAIN/VALIDATION/TEST SETS FOR CLASSIFICATION.
for i in `seq ${numsplits}`; do
  (
  if ${clobber} || [ ! -e ${data_output_dir}/split_${i}/data.pbtxt ]; then
    python split_reps.py ${data_output_dir}/dbn_all_layers/data.pbtxt \
      ${data_output_dir}/split_${i} ${prefix} ${i} ${gpu_mem} ${main_mem}
  fi
  )&
done
wait;

# DO LAYER-WISE CLASSIFICATION
for i in `seq ${numsplits}`
do
  #(
  for layer in image_input image_hidden1 image_hidden2 joint_hidden text_hidden2 text_hidden1 text_input
  do
    if ${clobber} || [ ! -e ${model_output_dir}/classifiers/split_${i}/${layer}_classifier_BEST ]; then
      echo Split ${i} ${layer}
      python ${trainer} models/classifiers/${layer}_classifier.pbtxt \
        trainers/classifiers/split_${i}.pbtxt eval.pbtxt || exit 1
    fi
  done
  #)&
done
#wait;
# COLLECT RESULTS AND PUT INTO A LATEX TABLE.
if ${clobber} || [ ! -e results.tex ]; then
  python create_results_table.py ${model_output_dir}/classifiers ${numsplits} \
    results.tex || exit 1
fi
