#!/bin/bash

# Modified from recipe for Mozilla Common Voice corpus v1
#
# Copyright 2017   Ewald Enzinger
# Apache 2.0

data=/mnt/e/zhuhan/data/cv_corpus_v1

. ./cmd.sh
. ./path.sh

stage=9

. ./utils/parse_options.sh

set -euo pipefail



if [ $stage -le 0 ]; then
  for part in valid-train valid-train-us-30000 valid-train-australia-2000 valid-train-england-2000 valid-train-indian-2000\
    valid-dev-us-1000 valid-dev-australia-1000 valid-dev-england-1000 valid-dev-indian-1000 \
    valid-test-us-1000 valid-test-australia-1000 valid-test-england-1000 valid-test-indian-1000; do
    # use underscore-separated names in data directories.
    local/data_prep.pl $data cv-$part data/$(echo $part | tr - _)
  done
  
  # Prepare ARPA LM and vocabulary using SRILM
  local/prepare_lm.sh data/valid_train
  # Prepare the lexicon and various phone lists
  # Pronunciations for OOV words are obtained using a pre-trained Sequitur model
  local/prepare_dict.sh

  # Prepare data/lang and data/local/lang directories
  utils/prepare_lang.sh data/local/dict \
    '<unk>' data/local/lang data/lang || exit 1

  utils/format_lm.sh data/lang data/local/lm.gz data/local/dict/lexicon.txt data/lang_test/
fi


if [ $stage -le 1 ]; then
  mfccdir=mfcc

  for part in valid_train_us_30000 valid_train_australia_2000 valid_train_england_2000 valid_train_indian_2000\
    valid_dev_us_1000 valid_dev_australia_1000 valid_dev_england_1000 valid_dev_indian_1000 \
    valid_test_us_1000 valid_test_australia_1000 valid_test_england_1000 valid_test_indian_1000; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 15 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
fi


if [ $stage -le 2 ]; then
  # combine train/dev/test data set separately.
  utils/combine_data.sh data/valid_train_all data/valid_train_us_30000 data/valid_train_australia_2000  data/valid_train_england_2000 data/valid_train_indian_2000
  utils/combine_data.sh data/valid_dev_all data/valid_dev_us_1000 data/valid_dev_australia_1000  data/valid_dev_england_1000 data/valid_dev_indian_1000
  utils/combine_data.sh data/valid_test_all data/valid_test_us_1000 data/valid_test_australia_1000  data/valid_test_england_1000 data/valid_test_indian_1000
  # Get the shortest 10000 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/valid_train_us_30000 10000 data/train_10kshort || exit 1;
  utils/subset_data_dir.sh data/valid_train_us_30000 20000 data/train_20k || exit 1;
fi

# train a monophone system
if [ $stage -le 3 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj 15 --cmd "$train_cmd" \
    data/train_10kshort data/lang exp/mono || exit 1;

  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph
  for testset in valid_test_us_1000; do
    steps/decode.sh --nj 15 --cmd "$decode_cmd" exp/mono/graph \
      data/$testset exp/mono/decode_$testset
  done

  steps/align_si.sh --boost-silence 1.25 --nj 15 --cmd "$train_cmd" \
    data/train_20k data/lang exp/mono exp/mono_ali_train_20k
fi

# train a first delta + delta-delta triphone system
if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_20k data/lang exp/mono_ali_train_20k exp/tri1

  # decode using the tri1 model
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
  for testset in valid_test_us_1000; do
    steps/decode.sh --nj 15 --cmd "$decode_cmd" exp/tri1/graph \
      data/$testset exp/tri1/decode_$testset
  done

  steps/align_si.sh --nj 15 --cmd "$train_cmd" \
    data/train_20k data/lang exp/tri1 exp/tri1_ali_train_20k
fi

# train an LDA+MLLT system.
if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_20k data/lang exp/tri1_ali_train_20k exp/tri2b

  # decode using the LDA+MLLT model
  utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
  for testset in valid_test_us_1000; do
    steps/decode.sh --nj 15 --cmd "$decode_cmd" exp/tri2b/graph \
      data/$testset exp/tri2b/decode_$testset
  done

fi


# train another LDA+MLLT system system on the entire training set.
if [ $stage -le 6 ]; then
  # Align utts using the tri2b model
  steps/align_si.sh --nj 15 --cmd "$train_cmd" --use-graphs true \
    data/valid_train_us_30000 data/lang exp/tri2b exp/tri2b_ali_valid_train_us_30000

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 4200 40000 \
    data/valid_train_us_30000 data/lang exp/tri2b_ali_valid_train_us_30000 exp/tri3b

  # decode using the new LDA+MLLT model
  utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph
  for testset in valid_test_us_1000 valid_test_australia_1000 valid_test_england_1000 valid_test_indian_1000; do
    steps/decode.sh --nj 15 --cmd "$decode_cmd" exp/tri3b/graph \
      data/$testset exp/tri3b/decode_$testset
  done

fi


if [ $stage -le 7 ]; then
  # Align all data for dnn training
  for ali_data in valid_train_all valid_dev_all valid_test_all; do
    steps/align_si.sh --nj 15 --cmd "$train_cmd" \
      data/$ali_data data/lang \
      exp/tri3b exp/tri3b_ali_$ali_data
  done
fi

if [ $stage -le 8 ]; then
  # combine train data set separately for multi-condition training.
  utils/combine_data.sh data/valid_train_us_australia data/valid_train_us_30000 data/valid_train_australia_2000
  utils/combine_data.sh data/valid_train_us_england data/valid_train_us_30000 data/valid_train_england_2000 
  utils/combine_data.sh data/valid_train_us_indian data/valid_train_us_30000 data/valid_train_indian_2000
fi


if [ $stage -le 9 ]; then
  # Align all data for dnn training
  for ali_data in valid_train_us_30000; do
    steps/align_si.sh --nj 15 --cmd "$train_cmd" \
      data/$ali_data data/lang \
      exp/tri3b exp/tri3b_ali_$ali_data
  done
fi

echo "Successfully Done !"
