#!/bin/bash

#SBATCH --job-name=train_gan_model
#SBATCH --output=train_gan_model_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --nodelist=cs004
#SBATCH --comment=intern_endding_soon
#SBATCH --mail-user=sudheer.tati@sjsu.edu
#SBATCH --requeue

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread


eval "$(conda shell.bash hook)"
conda activate main_3

# module load libsndfile-1.0.28-gcc-12.2.0-ax7udgw
# module load kenlm/2023.2


export FAIRSEQ_ROOT=/home/015910674/fairseq
export RVAD_ROOT=/home/015910674/rVADfast
export KENLM_ROOT=/opt/ohpc/pub/apps/kenlm/2023.2/build/bin
export HYDRA_FULL_ERROR=1 
export OC_CAUSE=1
export CUDA_LAUNCH_BLOCKING=1
export KALDI_ROOT=/home/015910674/projects/myASR/kaldi
export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.7/lib:$LD_LIBRARY_PATH
export PATH=$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH


PREFIX=w2v_unsup_gan_xp

# For wav2vec-U 2.0, use raw audio features
CONFIG_DIR=/home/015910674/fairseq/examples/wav2vec/unsupervised/config/gan/
CONFIG_NAME=w2vu2.yaml
TASK_DATA=/home/015910674/libri_speech_project/dataset/speech_features

# Unpaired text input
TEXT_DATA=/home/015910674/libri_speech_project/dataset/text_featrures_g2p/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=/home/015910674/libri_speech_project/dataset/text_featrures_g2p/phones/lm.phones.filtered.04.bin # KenLM 4-gram phoneme language model (LM data = GAN data here)

# PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
#     -m --config-dir config/gan \
#     --config-name $CONFIG_NAME \
#     --config-dir $CONFIG_DIR\
#     task.data=${TASK_DATA} \
#     task.text_data=${TEXT_DATA} \
#     task.kenlm_path=${KENLM_PATH} \
#     common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
#     model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
#     model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)'


PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX CUDA_LAUNCH_BLOCKING=1 fairseq-hydra-train \
    -m --config-dir config/gan \
    --config-name $CONFIG_NAME \
    --config-dir $CONFIG_DIR\
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2 model.gradient_penalty=1.5 \
    model.smoothness_weight=0.5 'common.seed=2'