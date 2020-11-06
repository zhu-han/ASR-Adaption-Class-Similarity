# kaldi

export KALDI_ROOT=/NasStore/5/speech-students/zhuhan/kaldi/kaldi-trunk-20200424

PATH=$PATH:$KALDI_ROOT/tools/openfst

PATH=$PATH:$KALDI_ROOT/src/featbin

PATH=$PATH:$KALDI_ROOT/src/gmmbin

PATH=$PATH:$KALDI_ROOT/src/bin

PATH=$PATH:$KALDI_ROOT/src/nnetbin



# for kaldi tools, such as srilm

source $KALDI_ROOT/tools/env.sh
