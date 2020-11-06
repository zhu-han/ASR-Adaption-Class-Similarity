#!/bin/bash

# Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Config:
nj=10
stage=0 # resume training with --stage=N
cmd=run.pl

. utils/parse_options.sh || exit 1;

# This script is made from the kaldi recipe of the 2nd CHiME Challenge Track 2
# made by Chao Weng

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <CHiME3 root directory>\n\n" `basename $0`
  echo "Please specifies a CHiME3 root directory"
  echo "If you use kaldi scripts distributed in the CHiME3 data,"
  echo "It would be `pwd`/../.."
  exit 1;
fi

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

# clean data
chime3_data=$1
wsj0_data=$chime3_data/data/WSJ0 # directory of WSJ0 in CHiME3. You can also specify your WSJ0 corpus directory

eval_flag=true # make it true when the evaluation data are released

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $stage -le 0 ]; then
  # process for clean speech and making LMs etc. from original WSJ0
  # note that training on clean data means original WSJ0 data only (no booth data)
  local/clean_wsj0_data_prep.sh $wsj0_data
  local/wsj_prepare_dict.sh
  utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang
  local/clean_chime3_format_data.sh
fi

if [ $stage -le 1 ]; then
  # process for close talking speech for real data (will not be used)
  # local/real_close_chime3_data_prep.sh $chime3_data

  # process for booth recording speech (will not be used)
  # local/bth_chime3_data_prep.sh $chime3_data

  # process for distant talking speech for real and simulation data
  local/real_noisy_chime3_data_prep.sh $chime3_data
  local/simu_noisy_chime3_data_prep.sh $chime3_data
fi

# Now make MFCC features for clean, close, and noisy data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
# clean data
if $eval_flag; then
  list="tr05_orig_clean dt05_orig_clean et05_orig_clean"
else
  list="tr05_orig_clean dt05_orig_clean"
fi
# real data
if $eval_flag; then
  list=$list" tr05_real_noisy dt05_real_noisy et05_real_noisy"
else
  list=$list" tr05_real_noisy dt05_real_noisy"
fi
# simulation data
if $eval_flag; then
  list=$list" tr05_simu_noisy dt05_simu_noisy et05_simu_noisy"
else
  list=$list" tr05_simu_noisy dt05_simu_noisy"
fi
mfccdir=mfcc
if [ $stage -le 2 ]; then
  for x in $list; do
    steps/make_mfcc.sh --nj 8 --cmd "$train_cmd" \
      data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
  done
fi

# make mixed training set from real and simulation training data
# multi = simu + real
if [ $stage -le 3 ]; then
  utils/combine_data.sh data/tr05_multi_noisy data/tr05_simu_noisy data/tr05_real_noisy
  utils/combine_data.sh data/dt05_multi_noisy data/dt05_simu_noisy data/dt05_real_noisy
  utils/combine_data.sh data/et05_multi_noisy data/et05_simu_noisy data/et05_real_noisy
fi

# training models for clean and noisy data
# if you want to check the performance of the ASR only using real/simu data
# please try to add "tr05_real_noisy" "tr05_simu_noisy"
# for train in tr05_multi_noisy tr05_real_noisy tr05_simu_noisy tr05_orig_clean; do
if [ $stage -le 4 ]; then
  for train in tr05_real_noisy tr05_orig_clean tr05_multi_noisy; do
    nspk=`wc -l data/$train/spk2utt | awk '{print $1}'`
    if [ $nj -gt $nspk ]; then
      nj2=$nspk
    else
      nj2=$nj
    fi
    # training monophone model
    steps/train_mono.sh --boost-silence 1.25 --nj $nj2 --cmd "$train_cmd" \
      data/$train data/lang exp/mono0a_$train
    steps/align_si.sh --boost-silence 1.25 --nj $nj2 --cmd "$train_cmd" \
      data/$train data/lang exp/mono0a_$train exp/mono0a_ali_$train

    # training triphone model with lad mllt features
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      2000 10000 data/$train data/lang exp/mono0a_ali_$train exp/tri1_$train
    steps/align_si.sh --nj $nj2 --cmd "$train_cmd" \
      data/$train data/lang exp/tri1_$train exp/tri1_ali_$train

    steps/train_lda_mllt.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      2500 15000 data/$train data/lang exp/tri1_ali_$train exp/tri2b_$train
    steps/align_si.sh  --nj $nj2 --cmd "$train_cmd" \
      --use-graphs true data/$train data/lang exp/tri2b_$train exp/tri2b_ali_$train

    steps/train_sat.sh --cmd "$train_cmd" \
      2500 15000 data/$train data/lang exp/tri2b_ali_$train exp/tri3b_$train
    utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri3b_$train exp/tri3b_$train/graph_tgpr_5k
  done
fi

# decoding
if [ $stage -le 5 ]; then
  for train in tr05_real_noisy tr05_orig_clean tr05_multi_noisy; do
    # if you want to know the result of the close talk microphone, please try the following
    # decode close speech
    # steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    #   exp/tri3b_$train/graph_tgpr_5k data/dt05_real_close exp/tri3b_$train/decode_tgpr_5k_dt05_real_close &
    # steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
    #   exp/tri3b_$train/graph_tgpr_5k data/et05_real_close exp/tri3b_$train/decode_tgpr_5k_et05_real_close &

    # decode real noisy speech
    steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
      exp/tri3b_$train/graph_tgpr_5k data/dt05_real_noisy exp/tri3b_$train/decode_tgpr_5k_dt05_real_noisy &
    steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
      exp/tri3b_$train/graph_tgpr_5k data/et05_real_noisy exp/tri3b_$train/decode_tgpr_5k_et05_real_noisy &
    wait
    # decode simu noisy speech
    steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
      exp/tri3b_$train/graph_tgpr_5k data/dt05_simu_noisy exp/tri3b_$train/decode_tgpr_5k_dt05_simu_noisy &
    steps/decode_fmllr.sh --nj 4 --num-threads 3 --cmd "$decode_cmd" \
      exp/tri3b_$train/graph_tgpr_5k data/et05_simu_noisy exp/tri3b_$train/decode_tgpr_5k_et05_simu_noisy &
    wait
  done
fi

# get the best scores
if [ $stage -le 6 ]; then
  #for train in tr05_multi_noisy tr05_real_noisy tr05_simu_noisy tr05_orig_clean; do
  for train in tr05_real_noisy tr05_orig_clean tr05_multi_noisy; do
    local/chime3_calc_wers.sh exp/tri3b_$train noisy > exp/tri3b_$train/best_wer_noisy.result
    head -n 15 exp/tri3b_$train/best_wer_noisy.result
  done
fi

# get new align
if [ $stage -le 7 ]; then
  for train_data in  tr05_real_noisy tr05_multi_noisy; do
    for ali_data in  tr05_multi_noisy dt05_multi_noisy et05_multi_noisy; do
      nspk=`wc -l data/$ali_data/spk2utt | awk '{print $1}'`
      if [ $nj -gt $nspk ]; then
        nj2=$nspk
      else
        nj2=$nj
      fi
      steps/align_fmllr.sh --nj $nj2 --cmd "$train_cmd" \
        data/${ali_data} data/lang exp/tri3b_${train_data} exp/tri3b_ali_${train_data}_${ali_data} || exit 1;
    done
  done
  for train_data in  tr05_orig_clean; do
    for ali_data in  tr05_orig_clean dt05_orig_clean et05_orig_clean; do
      nspk=`wc -l data/$ali_data/spk2utt | awk '{print $1}'`
      if [ $nj -gt $nspk ]; then
        nj2=$nspk
      else
        nj2=$nj
      fi
      steps/align_fmllr.sh --nj $nj2 --cmd "$train_cmd" \
        data/${ali_data} data/lang exp/tri3b_${train_data} exp/tri3b_ali_${train_data}_${ali_data} || exit 1;
    done
  done
fi

# Convert the noisy alignments to be consistent with the clean alignments.
if [ $stage -le 8 ]; then
  target_ali_dir=/mnt/d/zhuhan/kaldi-trunk/egs/chime3_modified/s5/exp/tri3b_tr05_orig_clean
  for train_data in  tr05_real_noisy tr05_multi_noisy; do
    for ali_data in  tr05_multi_noisy dt05_multi_noisy et05_multi_noisy; do
      olddir=exp/tri3b_ali_${train_data}_${ali_data}
      newdir=exp/tri3b_converted_ali_${train_data}_${ali_data}
      echo "$0: Converting alignments from $olddir to $newdir"
      cp -rf $olddir $newdir || exit 1;
      cp $target_ali_dir/final.mdl $newdir/final.mdl && cp $target_ali_dir/tree $newdir/tree || exit 1;
      nj=`cat $olddir/num_jobs`
      $cmd JOB=1:$nj $newdir/log/convert.JOB.log \
        convert-ali  $olddir/final.mdl $newdir/final.mdl $newdir/tree \
        "ark:gunzip -c $olddir/ali.JOB.gz|" "ark:|gzip -c >$newdir/ali.JOB.gz" || exit 1;
    done
  done
fi


# make mixed training set and alignments from real noisy and ori clean training data (for multi-style dnn training)
# multi_noisy_clean = real_noisy + orig_clean
if [ $stage -le 9 ]; then
  utils/combine_data.sh data/tr05_multi_noisy_clean data/tr05_real_noisy data/tr05_orig_clean
  multi_alidir=exp/tri3b_ali_tr05_multi_noisy_clean
  clean_alidir=exp/tri3b_ali_tr05_orig_clean_tr05_orig_clean
  noisy_alidir=exp/tri3b_converted_ali_tr05_real_noisy_tr05_multi_noisy
  cp -rf $clean_alidir $multi_alidir
  num_jobs_clean=`cat $clean_alidir/num_jobs`
  num_jobs_noisy=`cat $noisy_alidir/num_jobs`
  num_jobs_multi=`expr $num_jobs_clean + $num_jobs_noisy`
  
  noisy_index=1
  while [ $noisy_index -le $num_jobs_noisy ]
  do
    multi_index=`expr $noisy_index + $num_jobs_clean`
    cp $noisy_alidir/ali.${noisy_index}.gz $multi_alidir/ali.${multi_index}.gz
    noisy_index=`expr $noisy_index + 1`
  done
  echo $num_jobs_multi > $multi_alidir/num_jobs
fi


# make mixed dev set and alignments from real noisy and ori clean dev data (for multi-style dnn dev)
# multi_noisy_clean = real_noisy + orig_clean
if [ $stage -le 13 ]; then
  utils/combine_data.sh data/dt05_multi_noisy_clean data/dt05_real_noisy data/dt05_orig_clean
  multi_alidir=exp/tri3b_ali_dt05_multi_noisy_clean
  clean_alidir=exp/tri3b_ali_tr05_orig_clean_dt05_orig_clean
  noisy_alidir=exp/tri3b_converted_ali_tr05_multi_noisy_dt05_multi_noisy
  cp -rf $clean_alidir $multi_alidir
  num_jobs_clean=`cat $clean_alidir/num_jobs`
  num_jobs_noisy=`cat $noisy_alidir/num_jobs`
  num_jobs_multi=`expr $num_jobs_clean + $num_jobs_noisy`
  
  noisy_index=1
  while [ $noisy_index -le $num_jobs_noisy ]
  do
    multi_index=`expr $noisy_index + $num_jobs_clean`
    cp $noisy_alidir/ali.${noisy_index}.gz $multi_alidir/ali.${multi_index}.gz
    noisy_index=`expr $noisy_index + 1`
  done
  echo $num_jobs_multi > $multi_alidir/num_jobs
fi

if [ $stage -le 10 ]; then
  for x in $list; do
    feat-to-len scp:data/$x/feats.scp ark,t:data/$x/feats.lengths
  done
fi

# make mixed training set and alignments from multi_noisy_clean and simu noisy training data (for GRL dnn training).
# note that the alignment for simu noisy is not really used during training, 
# it is used as a psedu label, just for the regular process of label and feature to be successful.
# all = multi_noisy_clean + simu noisy
if [ $stage -le 11 ]; then
  utils/combine_data.sh data/tr05_all data/tr05_multi_noisy_clean data/tr05_simu_noisy
  # the alignment of multi_alidir already contain the label of simu noisy (use the model trained using real noisy only)

fi


echo "`basename $0` Done."
