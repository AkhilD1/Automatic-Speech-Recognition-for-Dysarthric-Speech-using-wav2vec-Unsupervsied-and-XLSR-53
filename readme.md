# Applications of the Unsupervised ASR Technology in Different Domains

Automatic Speech Recognition (ASR) is a technology that converts speech signals into text or commands. Traditional ASR models rely on labeled audio data of healthy speech, which poses a challenge when dealing with impaired speech due to difficulties in acquiring labeled data. This has impeded the progress of ASR technology in this domain. To address this issue, this project explores the use of both self-supervised and unsupervised learning approaches in developing an ASR sys- tem for impaired speech. The self-supervised approach, such as wav2vec2.0, has shown promising results, with training using only one hour of labeled data outperforming the previous state-of-the-art models that require 100 times more labeled data. In addition to self-supervised learning, the project also aims to build an unsupervised ASR on dysarthric speech using wav2vec-Unsupervised. This method uses a GAN to map the unlabeled audio and text, and building an unsupervised ASR for dysarthric speech is a significant challenge due to the inadequacy of data. Based on our experiments, we obtained the best Word Error Rate (WER) of 14.8% using the self- supervised model and 55.9% using the unsupervised model when trained on our custom datasets created by combining multiple open-source datasets

## Dataset Info:

## UA-speech
It is an audio dataset of people having Neuromotor disorders, widely used for the development of Automatic Speech Recognition (ASR) models. It also contains video of recordings, which shows that the audio is captured using an eight-microphone array and preamplifiers. These will not be used in our project. It contains audio data from 15 dysarthric speakers with various levels of dysarthria. It consists of single and isolated words along with audio from 13 normal speakers. The words consist of digits, the international radio alphabet, word processing commands, the most common words, and 300 words from Gutenberg novels. All of these count to 765 words and the recordings are done over 3 sessions.

## TORGO Dataset
One of the most often used databases for people with dysarthric articulation is the TORGO dataset from the University of Toronto. Eight speakers (five men and three women), all of whom had either cerebral palsy or amyotrophic lateral sclerosis, were heard on the audio. The dataset also contains audio recordings of 7 standard speakers (4 male speakers and 3 female speakers), which can be utilized as a control.

Torgo Dataset: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

## Libri-Speech Dataset
LibriSpeech is a large-scale corpus of English speech recordings, designed for speech recognition and related research fields. It contains over 1000 hours of high-quality audio data, transcribed at the level of individual words and phrases. The dataset contains 2,456 audiobooks, with a total of 16,000 speakers and 460,000 unique English words. The recordings were made at a sampling rate of 16 kHz and are provided as WAV files. The recordings are drawn from diverse sources, including audiobooks, lectures, and public domain materials, and cover a wide range of topics and speaking styles. LibriSpeech has become a popular benchmark for evaluating the performance of speech recognition algorithms and is widely used in academic and industrial research.  For our project we are using 100 hours of data from Libri-Speech Database.

Libri-speech dataset: https://www.openslr.org/resources/12/train-clean-100.tar.gz

# Project Environment Setup:
## Requirements and Installation
* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

#### Install Fairseq library

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

* set FAIRSEQ_ROOT environmental variable to your fairseq installation
* set RVAD_ROOT environmental variable to a checkout of [rVADfast](https://github.com/zhenghuatan/rVADfast)
* set KENLM_ROOT environmental variable to the location of [KenLM](https://github.com/kpu/kenlm) binaries
* install [PyKaldi](https://github.com/pykaldi/pykaldi) and set KALDI_ROOT environmental variable to the location of your kaldi installation. To use the version bundled with PyKaldi, you can use /path/to/pykaldi/tools/kaldi

**For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## Format Data and Preprocess the Audio and Text data.

* In this project preprocessing and training are done in the hpc. Please connect to HPC and excute the commands using slurm.

* Copy all the audio files from these datasets into a single folder. This step ensures that all the audio data is consolidated and easily accessible during training.

* Set the `DATASET_PATH` environment variable after installing the necessary dependencies. This variable will point to the folder where the audio files are stored. Make sure to provide the correct path to the folder containing the audio data.

* Copy all the text data into a single file. This consolidated text data will be used as the training data for the GAN discriminator. Ensure that the text is in the appropriate format for further processing.

Create new audio files without silences:
```shell
# create a manifest file for the set original of audio files
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py $DATASET_PATH --ext wav --dest /path/to/new/train.tsv --valid-percent 0

python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/vads.py -r $RVAD_ROOT < /path/to/train.tsv > train.vads

python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py --tsv /path/to/train.tsv --vads train.vads --out $DATASET_PATH

python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py $DATASET_PATH --ext wav --dest /path/to/new/train.tsv --valid-percent 0.01
```
Run the above command or run using the sbatch files
```shell
sbatch preprocess_text_audio_data.sh
```
update the commands in the sbatch file accordingly.

Next, we need to preprocess the audio data to better match phonemized text data:

```shell
# wav2vec-U
zsh $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh $DATASET_PATH /output/dir /path/to/wav2vec2/model.pt 512 14
# wav2vec-U 2.0
zsh $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio_v2.sh $DATASET_PATH /output/dir /path/to/wav2vec2/model.pt 64 14
```

Now we need to prepare text data:
```shell
zsh $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_text.sh en /path/to/text/file /output/dir 1000 espeak /path/to/fasttext/lid/model 0.25
```

The fourth argument is minimum number observations of phones to keep. If your text corpus is small, you might want to reduce this number.

The fifth argument is which phonemizer to use. Supported values are [espeak](http://espeak.sourceforge.net/), [espeak-ng](https://github.com/espeak-ng/espeak-ng), and [G2P](https://github.com/Kyubyong/g2p) (english only).

Pre-trained fasttext LID models can be downloaded [here](https://fasttext.cc/docs/en/language-identification.html).

The last argument is the probability to introduce silence (`<SIL>`) between the word boundaries. We found the value `0.25`/`0.5` works in general for wav2vec-U and the 2.0  version respectively, but you might want to vary for languages that are never tested.



## Generative adversarial training (GAN)

We then use a GAN model to build a first unsupervised ASR model. The data preparation above of both speech features and text data is a necessary procedure that enables the generator to match speech to text in an unsupervised way. 

Launching GAN training on top of preprocessed features, with default hyperparameters can be done with:

Training the wav2vec-U model on the UA-speech dataset
```shell
sbatch train_wav2vec_ua_speech.sh
```
update the commands in the sbatch file accordingly.

Training the wav2vec-u 2.0 on the UA-speech + Torgo + libiri-speech dataset
```shell
sbatch train_wav2vec_gpu_usage.sh
```

Training the wav2vec-u 2.0 on the UA-speechdataset
```shell
sbatch train_wav2vec.sh
```
update the commands in the sbatch file accordingly.

Once we find the best checkpoint (chosen using unsupervised metric that combined language model perplexity and vocabulary usage), we can use it to generate phone labels (or word labels with an appropriate kaldi WFST):

```shell
python w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=/path/to/dir/with/features \
fairseq.common_eval.path=/path/to/gan/checkpoint \ 
fairseq.dataset.gen_subset=valid results_path=/where/to/save/transcriptions
```

The decoding without LM works best on the same adjacent-mean-pooled features that the gan was trained on, while decoding with LM works better on features before the adjacent timestep mean-pooling step (without the "_pooled" suffix).
