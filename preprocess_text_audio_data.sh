#!/bin/bash

#SBATCH --job-name=speech_features_extraction
#SBATCH --output=step_2_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-user=sudheer.tati@sjsu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread


eval "$(conda shell.bash hook)"
conda activate main_3

export FAIRSEQ_ROOT=/home/015910674/fairseq
export RVAD_ROOT=/home/015910674/rVADfast

# Define the range of input files to process
# SBATCH --array=15-17

module load libsndfile-1.0.28-gcc-12.2.0-ax7udgw
module load kenlm/2023.2

export PATH="$HOME/zsh/usr/sbin:$HOME/zsh/usr/bin:$HOME/zsh/bin:$PATH"
export MANPATH="$HOME/zsh/usr/share/man:$MANPATH"
L='/lib:/lib64:/usr/lib:/usr/lib64'
export LD_LIBRARY_PATH="$HOME/zsh/usr/lib:$HOME/centos/usr/lib64:$L"
export KENLM_ROOT=/opt/ohpc/pub/apps/kenlm/2023.2/build/bin
export KALDI_ROOT=/home/015910674/projects/myASR/kaldi
export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.7/lib:$LD_LIBRARY_PATH
export PATH=$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH

echo $SLURM_ARRAY_TASK_ID
module load libsndfile-1.0.28-gcc-12.2.0-ax7udgw
module load libsndfile-1.0.28-gcc-12.2.0-ax7udgw
# Run the command with different input files for each task
# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/vads.py -r $RVAD_ROOT < /home/015910674/uaspeech_project/split_train_files/train_$SLURM_ARRAY_TASK_ID.tsv > /home/015910674/uaspeech_project/split_train_files/train_$SLURM_ARRAY_TASK_ID.vads
# python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /home/015910674/libri_speech_project/dataset/processed_audios --ext flac --dest /home/015910674/libri_speech_project/dataset/processed_audios/tsv_files --valid-percent 0.01
# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py --tsv /home/015910674/libri_speech_project/dataset/train_$SLURM_ARRAY_TASK_ID.tsv  --vads /home/015910674/libri_speech_project/dataset/train_$SLURM_ARRAY_TASK_ID.vads --out /home/015910674/libri_speech_project/dataset/processed_audios/train_$SLURM_ARRAY_TASK_ID
# $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio_v2.sh /home/015910674/libri_speech_project/dataset/processed_audios/tsv_files /home/015910674/libri_speech_project/dataset/speech_features /home/015910674/projects/xlsr_53_56k.pt 64 14
# z# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py /home/015910674/libri_speech_project/dataset/processed_audios/tsv_files --split "valid" --save-dir /home/015910674/libri_speech_project/dataset/speech_features --checkpoint /home/015910674/projects/xlsr_53_56k.pt --layer 14
$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh /home/015910674/projects/processed_audios/train.tsv /home/015910674/projects/speech_features_v1 /home/015910674/projects/xlsr_53_56k.pt 512 14

# zsh $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh /home/015910674/projects/processed_audios/train.tsv /home/015910674/projects/speech_features_v1_4 /home/015910674/projects/xlsr_53_56k.pt 512 14

#### python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py /home/015910674/projects/processed_audios/train.tsv  --split train --save-dir /home/015910674/projects/speech_features_v1_3 --checkpoint /home/015910674/projects/w2v_large_lv_fsh_swbd_cv_ftls960_updated.pt --layer 14
# python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /home/015910674/uaspeech_project/dataset/UASpeech --ext wav --dest /home/015910674/uaspeech_project/dataset/UASpeech/tsv_files --valid-percent 0


# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py --tsv /home/015910674/projects/train_$SLURM_ARRAY_TASK_ID.tsv  --vads /home/015910674/projects/rvads/train_$SLURM_ARRAY_TASK_ID.vads --out /home/015910674/projects/processed_audios/train_$SLURM_ARRAY_TASK_ID

# # python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /home/015910674/projects/processed_audios --ext wav --dest /home/015910674/projects/processed_audios/train.tsv --valid-percent 0.01

# $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio_v2.sh /home/015910674/projects/processed_audios/train.tsv /home/015910674/projects/step_2 /home/015910674/projects/xlsr_53_56k.pt 64 14

# # python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py /home/015910674/projects/step_2/mfcc train 1 /home/015910674/projects/step_2/mfcc/cls64 128 --percent -1

$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_text.sh en /home/015910674/projects/text_data_processed.txt /home/015910674/projects/text_data 1000 espeak /home/015910674/projects/lid.176.bin 0.25

# # python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py --lang en --fasttext-model /home/015910674/projects/lid.176.bin < /home/015910674/projects/text_data.txt | grep -v '\-\-\-' >! /home/015910674/projects/text_data/lm.upper.lid.txt

# python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones.txt --only-source --destdir $target_dir/phones --thresholdsrc 10 --padding-factor 1 --dict-only

# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0.25 --surround --lexicon $target_dir/lexicon_filtered.lst < $target_dir/lm.upper.lid.txt > $target_dir/phones/lm.phones.filtered.txt

# lg="en-us" python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words_sil lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"

# lg="en-us" python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words_sil lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"


# lg="en-us" python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn


# $KENLM_ROOT/lmplz -o 4 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback > $target_dir/phones/lm.phones.filtered.04.arpa
# $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.04.arpa $target_dir/phones/lm.phones.filtered.04.bin
# $KENLM_ROOT/lmplz -o 6 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback > $target_dir/phones/lm.phones.filtered.06.arpa
# $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.06.arpa $target_dir/phones/lm.phones.filtered.06.bin

# lg="en-us" python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil lm_arpa=$target_dir/phones/lm.phones.filtered.06.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"


# # python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref /home/015910674/projects/text_data/lm.upper.lid.txt --only-source --destdir /home/015910674/projects/text_data --thresholdsrc 2 --padding-factor 1 --dict-only

# # python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref /home/015910674/projects/text_data/lm.upper.lid.txt --only-source --destdir /home/015910674/projects/text_data --thresholdsrc 2 --padding-factor 1 --dict-only

# # python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py --compact < /home/015910674/projects/text_data/words.txt > /home/015910674/projects/text_data/phones.txt

# # one=$(echo "1" |  phonemize -p ' ' -w '' -l en --language-switch remove-flags)


# # sed 's/$/ 1/' /home/015910674/projects/text_data/words.txt | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -o /home/015910674/projects/text_data/phones.txt -p ' ' -w '' -l en -j 70 --language-switch remove-flags

# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d $target_dir/phones/dict.txt < $target_dir/lexicon.lst > $target_dir/lexicon_filtered.lst

# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0.25 --surround --lexicon $target_dir/lexicon_filtered.lst < $target_dir/lm.upper.lid.txt > $target_dir/phones/lm.phones.filtered.txt

# python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d $target_dir/phones/dict.txt < $target_dir/lexicon.lst > $target_dir/lexicon_filtered.lst


# PREFIX=w2v_unsup_gan_xp

# # For wav2vec-U 2.0, use raw audio features
# CONFIG_NAME=w2vu2
# TASK_DATA=/path/to/features/

# # Unpaired text input
# TEXT_DATA=/path/to/data/phones  # path to fairseq-preprocessed GAN data (phones dir)
# KENLM_PATH=/path/to/data/phones/kenlm.phn.o4.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

# PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
#     -m --config-dir config/gan \
#     --config-name $CONFIG_NAME \
#     task.data=${TASK_DATA} \
#     task.text_data=${TEXT_DATA} \
#     task.kenlm_path=${KENLM_PATH} \
#     common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
#     model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
#     model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)'

# $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio_v2.sh /home/015910674/projects/processed_audios/train.tsv /home/015910674/projects/step_4 /home/015910674/projects/xlsr_53_56k.pt 64 14

# python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py /home/015910674/projects/step_2 train 1 0 /home/015910674/projects/step_2/mfcc

# python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py /home/015910674/projects/step_2/mfcc train /home/015910674/projects/step_2/mfcc/cls64 1 0 /home/015910674/projects/step_2/mfcc/cls64_idx

# cp /home/015910674/projects/step_2/mfcc/cls64_idx/train_0_1.km /home/015910674/projects/step_2/train.km

#  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py  /home/015910674/projects/processed_audios/train.tsv --split "train" --save-dir /home/015910674/projects/step_4 --checkpoint /home/015910674/projects/xlsr_53_56k.pt --layer 14

#  python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py /home/015910674/projects/step_5/mfcc train 1 /home/015910674/projects/step_5/mfcc/cls64 128 --percent -1


# $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio_v2.sh /home/015910674/projects/processed_audios/train.tsv /home/015910674/projects/step_7 /home/015910674/projects/xlsr_53_56k.pt 64 14

# python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py /home/015910674/projects/step_5/mfcc train /home/015910674/projects/step_5/mfcc/cls64 1 0 /home/015910674/projects/step_5/mfcc/cls64_idx

# cp /home/015910674/projects/step_5/mfcc/cls64_idx/train_0_1.km /home/015910674/projects/step_5/train.km

# python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /home/015910674/projects/Audio\ data/UASpeech --ext wav --dest /home/015910674/projects/processed_audio_data_2/ --valid-percent 0

python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py --config-dir /home/015910674/fairseq/examples/wav2vec/unsupervised/config/generate --config-name viterbi fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised fairseq.task.data=/home/015910674/projects/speech_features_v1/precompute_pca512_cls128_mean_pooled fairseq.common_eval.path=/home/015910674/checkpoint_ua_speech_g2p_small_sil/0/checkpoint_best.pt fairseq.dataset.gen_subset=valid results_path=/home/015910674/ua_speech_results_sil

# python /home/015910674/fairseq/examples/wav2vec/libri_labels.py /home/015910674/libri_speech_project/dataset/LibriSpeech/train-clean-100/tsv_files/train.tsv --output-dir /home/015910674/libri_speech_project/dataset/LibriSpeech/train-clean-100/tsv_files/text_data --output-name train

zsh $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_text.sh en /home/015910674/projects/ua_speech_text_data/text_data.txt /home/015910674/projects/ua_speech_text_data/text_features 20 G2P /home/015910674/projects/lid.176.bin 0.25

zsh $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh /home/015910674/projects/processed_audios/train.tsv /home/015910674/projects/speech_features_v1 /home/015910674/projects/xlsr_53_56k.pt 512 14


fairseq-hydra-train \
    distributed_training.distributed_port=4002 \
    task.data=/home/015910674/wav2vec_project/ua_speech/split_data \
    model.w2v_path=/home/015910674/projects/xlsr_53_56k.pt \
    --config-dir /home/015910674/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_100h

python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /home/015910674/projects/Audio\ data/TORGO_DATASET --ext wav --dest /home/015910674/wav2vec_project/torgo --valid-percent 0


python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py --config-dir /home/015910674/fairseq/examples/wav2vec/unsupervised/config/generate --config-name viterbi fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised fairseq.task.data=/home/015910674/projects/speech_features_v1/precompute_pca512_cls128_mean_pooled fairseq.common_eval.path=/home/015910674/checkpoint_ua_speech_g2p_small_sil/0/checkpoint_best.pt fairseq.dataset.gen_subset=valid results_path=/home/015910674/ua_speech_results_sil
