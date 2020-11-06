#!/bin/bash

#  Copyright  2015  Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
#  Apache 2.0.

set -e

rdir=$1

if [ -e $rdir ]; then

    lmw=`cat $rdir/scoring_kaldi/best_wer | grep -o 'wer_.*_.*' | awk -F_ '{print $2}'`
    penalty=`cat $rdir/scoring_kaldi/best_wer | grep -o 'wer_.*_.*' | awk -F_ '{print $3}'`

    for a in _BUS _CAF _PED _STR; do
        grep $a $rdir/scoring_kaldi/test_filt.txt > $rdir/scoring_kaldi/test_filt_${a}.txt

        cat $rdir/scoring_kaldi/penalty_$penalty/${lmw}.txt |  sed s:\<UNK\>::g \
        | compute-wer --text --mode=present ark:$rdir/scoring_kaldi/test_filt_${a}.txt ark,p:- \
        1> $rdir/scoring_kaldi/${a}_wer_${lmw}_${penalty} 2> /dev/null
    done
    echo -n "${decoding}_${output} WER: `grep WER $rdir/wer_${lmw}_${penalty} | cut -f 2 -d" "`% (Average), "
    echo -n "`grep WER $rdir/scoring_kaldi/_BUS_wer_${lmw}_${penalty} | cut -f 2 -d" "`% (BUS), "
    echo -n "`grep WER $rdir/scoring_kaldi/_CAF_wer_${lmw}_${penalty} | cut -f 2 -d" "`% (CAFE), "
    echo -n "`grep WER $rdir/scoring_kaldi/_PED_wer_${lmw}_${penalty} | cut -f 2 -d" "`% (PEDESTRIAN), "
    echo -n "`grep WER $rdir/scoring_kaldi/_STR_wer_${lmw}_${penalty} | cut -f 2 -d" "`% (STREET)"
    echo ""
    echo "-------------------"
fi
