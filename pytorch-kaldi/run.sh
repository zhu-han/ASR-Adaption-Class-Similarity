#!/bin/bash


set -e
set -u
set -o pipefail


    ###########################################
    ###########################################
    ###########################################
    #               CHIME3                    #
    ###########################################
    ###########################################
    ###########################################


    ############################################################
    ############################################################
    #                  train source model                      #
    ############################################################
    ############################################################

    if [ ! -f kaldi_decoding_scripts/local/wer_output_filter ]; then
        if [ -f kaldi_decoding_scripts/local/wsj_wer_output_filter ];then
            mv kaldi_decoding_scripts/local/wsj_wer_output_filter kaldi_decoding_scripts/local/wer_output_filter
        else
            echo "kaldi_decoding_scripts/local/wer_output_filter not exist, which is needed for scoring" && exit 1;
        fi
    fi
    # train model using only source domain data
    python -u run_exp_modified.py --train "0,1,2,3,4,5" cfg/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_valid_dt05_orig_clean_test_ori_real_lr_4e-4_epoch_30.cfg >log/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_valid_dt05_orig_clean_test_ori_real_lr_4e-4_epoch_30.log 2>&1

    python -u run_exp_modified.py --decode "0,1,2,3,4,5" cfg/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_valid_dt05_orig_clean_test_ori_real_lr_4e-4_epoch_30.cfg >>log/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_valid_dt05_orig_clean_test_ori_real_lr_4e-4_epoch_30.log 2>&1

    ############################################################
    ############################################################
    #                  train target model                      #
    ############################################################
    ############################################################

    # train model using only target domain data
    python -u run_exp_modified.py --train "0" cfg/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_30.cfg >log/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_30.log 2>&1
    
    python -u run_exp_modified.py --decode "0" cfg/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_30.cfg >>log/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_30.log 2>&1

    ############################################################
    ############################################################
    #                  source + target model                   #
    ############################################################
    ############################################################

    # train model using source + target domain data
    python -u run_exp_modified.py --train "0,1" cfg/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_multi_noisy_clean_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_30_new.cfg >log/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_multi_noisy_clean_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_30_new.log 2>&1
	python -u run_exp_modified.py --decode "0,1" cfg/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_multi_noisy_clean_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_30_new.cfg >>log/chime3/train/chime3_GRU_4layer_mfcc_train_tr05_multi_noisy_clean_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_30_new.log 2>&1



    ############################################################
    ############################################################
    #                       fine-tune                          #
    ############################################################
    ############################################################


    # forward and create soft label embedding

    ## forward using source domain model and source domain data
    python -u run_exp_modified.py --production cfg/chime3/forward/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_forward_tr05_orig_clean.cfg >log/chime3/forward/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_forward_tr05_orig_clean.log 2>&1

    ## create soft label embedding
    alidir=/nobackup/bx/zhuhan/kaldi/kaldi-trunk-20181127/egs/chime3_modified/s5/exp/tri3b_ali_tr05_orig_clean_tr05_orig_clean
    forwarddir=/nobackup/bx/zhuhan/pytorch-kaldi/exp/chime3/forward/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_forward_tr05_orig_clean/

    python -u local/create_soft_label_embedding.py --forward-dir $forwarddir --ali-dir $alidir --temp 5 --nj 5 >> log/chime3/forward/create_soft_label_embedding.log 2>&1
    python -u local/create_soft_label_embedding.py --forward-dir $forwarddir --ali-dir $alidir --temp 2 --nj 5 >> log/chime3/forward/create_soft_label_embedding.log 2>&1
    python -u local/create_soft_label_embedding.py --forward-dir $forwarddir --ali-dir $alidir --temp 1 --nj 5 >> log/chime3/forward/create_soft_label_embedding.log 2>&1


    ###########################################
    # fine-tune based on the source GRU model
    ###########################################

    ## 1600 utt

    ### lr = 4e-4
    python -u run_exp_modified.py --train "0" cfg/chime3/fine-tune/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20.cfg >log/chime3/fine-tune/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20.log 2>&1
    
    python -u run_exp_modified.py --decode "0" cfg/chime3/fine-tune/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20.cfg >>log/chime3/fine-tune/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20.log 2>&1

    #######################
    # soft fine-tune
    #######################

    # 1600 utt

    ## temp_1
    python -u run_exp_modified.py --train "0" cfg/chime3/soft_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1.cfg > log/chime3/soft_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/soft_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1.cfg >> log/chime3/soft_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1.log 2>&1

    ## temp_2
    python -u run_exp_modified.py --train "0" cfg/chime3/soft_fine-tune/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2.cfg > log/chime3/soft_fine-tune/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/soft_fine-tune/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2.cfg >> log/chime3/soft_fine-tune/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2.log 2>&1

    ## temp_5
    python -u run_exp_modified.py --train "0" cfg/chime3/soft_fine-tune/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5.cfg > log/chime3/soft_fine-tune/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/soft_fine-tune/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5.cfg >> log/chime3/soft_fine-tune/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5.log 2>&1



    #######################
    # soft hard fine-tune
    #######################

    # 1600 utt

    ## temp_1

    ### weight_0.1

    python -u run_exp_modified.py --train "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg > log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1
    
    python -u run_exp_modified.py --decode "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >> log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1

    ### weight_0.2

    python -u run_exp_modified.py --train "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg > log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1
    
    python -u run_exp_modified.py --decode "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >> log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1

    ### weight_0.5

    python -u run_exp_modified.py --train "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg > log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1
    
    python -u run_exp_modified.py --decode "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >> log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1


    ### weight_1

    python -u run_exp_modified.py --train "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_1.cfg > log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1
    
    python -u run_exp_modified.py --decode "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_1.cfg >> log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1

    ### weight_2

    python -u run_exp_modified.py --train "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_2.cfg > log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_2.cfg >> log/chime3/soft_hard_fine-tune/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_2.log 2>&1


    ## temp_2

    ### weight_1

    python -u run_exp_modified.py --train "0" cfg/chime3/soft_hard_fine-tune/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_1.cfg > log/chime3/soft_hard_fine-tune/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1
    
    python -u run_exp_modified.py --decode "0" cfg/chime3/soft_hard_fine-tune/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_1.cfg >> log/chime3/soft_hard_fine-tune/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1


    ## temp_5

    ### weight_1


    python -u run_exp_modified.py --train "0" cfg/chime3/soft_hard_fine-tune/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_1.cfg > log/chime3/soft_hard_fine-tune/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1
    
    python -u run_exp_modified.py --decode "0" cfg/chime3/soft_hard_fine-tune/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_1.cfg >> log/chime3/soft_hard_fine-tune/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_soft_hard_fine-tune_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1


    ###############
    # distillation
    ###############
    # 1600 utt

    ## temp = 1


    ### weight_0.1
    python -u run_exp_modified.py --train "0"  cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >>log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0"  cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >>log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0"  cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >>log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1


    ### weight_1
    python -u run_exp_modified.py --train "0"  cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_1.cfg >log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_1.cfg >>log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1

    ### weight_2

    python -u run_exp_modified.py --train "0"  cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_2.cfg >log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_2.cfg >>log/chime3/distillation/temp_1/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_1_weight_2.log 2>&1


    ## temp = 2

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg >log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg >>log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1 


    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg >log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg >>log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg >log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg >>log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_1.cfg >log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_1.cfg >>log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1

    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_2.cfg >log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_2.cfg >>log/chime3/distillation/temp_2/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_2_weight_2.log 2>&1

    ## temp = 5

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg >log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg >>log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg >log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg >>log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg >log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg >>log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_1.cfg >log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_1.cfg >>log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1

    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_2.cfg >log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_2.cfg >>log/chime3/distillation/temp_5/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_distillation_tr05_real_noisy_valid_dt05_real_noisy_test_real_lr_4e-4_epoch_20_temp_5_weight_2.log 2>&1



    #######################
    # compute frame accuracy
    #######################

    python -u run_exp_modified.py --train "0" cfg/chime3/compute_frame_acc/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_compute_frame_err_tr05_real_noisy_valid_tr05_real_noisy_test_real_lr_4e-4_epoch_1.cfg > log/chime3/compute_frame_acc/chime3_GRU_4layer_mfcc_train_tr05_orig_clean_compute_frame_err_tr05_real_noisy_valid_tr05_real_noisy_test_real_lr_4e-4_epoch_1.log




    ###########################################
    ###########################################
    ###########################################
    #              CommonVoice                #
    ###########################################
    ###########################################
    ###########################################


    ############################################################
    ############################################################
    #                  train source model                      #
    ############################################################
    ############################################################

    if [ -f kaldi_decoding_scripts/local/wer_output_filter ]; then
        mv kaldi_decoding_scripts/local/wer_output_filter kaldi_decoding_scripts/local/wsj_wer_output_filter
    fi
    # train model using only source domain data
    python -u run_exp_modified.py --train "0,1,2,3,4,5" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_30000_valid_us_1000_test_us_1000_australia_1000_england_1000_indian_1000_lr_4e-4_epoch_30.cfg > log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_30000_valid_us_1000_test_us_1000_australia_1000_england_1000_indian_1000_lr_4e-4_epoch_30.log 2>&1
    
    python -u run_exp_modified.py --decode "0,1,2,3,4,5" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_30000_valid_us_1000_test_us_1000_australia_1000_england_1000_indian_1000_lr_4e-4_epoch_30.cfg >> log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_30000_valid_us_1000_test_us_1000_australia_1000_england_1000_indian_1000_lr_4e-4_epoch_30.log 2>&1


    ############################################################
    ############################################################
    #                  train target model                      #
    ############################################################
    ############################################################

    # train model using only target domain data

    # australia

    python -u run_exp_modified.py --train "0" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_30.cfg > log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_30.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_30.cfg >> log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_30.log 2>&1

    # england
    
    python -u run_exp_modified.py --train "0" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_30.cfg > log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_30.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_30.cfg >> log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_30.log 2>&1


    # indian
    
    python -u run_exp_modified.py --train "0" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_30.cfg > log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_30.log 2>&1
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_30.cfg >> log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_30.log 2>&1


    ############################################################
    ############################################################
    #                  source + target model                   #
    ############################################################
    ############################################################

    # train model using source + target domain data

    # australia

    python -u run_exp_modified.py --train "0,1" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_australia_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_30.cfg > log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_australia_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_30.log 2>&1
	python -u run_exp_modified.py --decode "0,1" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_australia_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_30.cfg >> log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_australia_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_30.log 2>&1

    # england
    
    python -u run_exp_modified.py --train "0,1" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_england_valid_england_1000_test_england_1000_lr_4e-4_epoch_30.cfg > log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_england_valid_england_1000_test_england_1000_lr_4e-4_epoch_30.log 2>&1
	python -u run_exp_modified.py --decode "0,1" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_england_valid_england_1000_test_england_1000_lr_4e-4_epoch_30.cfg >> log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_england_valid_england_1000_test_england_1000_lr_4e-4_epoch_30.log 2>&1

    # indian
    
    python -u run_exp_modified.py --train "0,1" cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_indian_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_30.cfg > log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_indian_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_30.log 2>&1
    python -u run_exp_modified.py --decode "0,1"cfg/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_indian_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_30.cfg >> log/commonvoice/train/commonvoice_GRU_4layer_mfcc_train_us_indian_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_30.log 2>&1



    ############################################################
    ############################################################
    #                       fine-tune                          #
    ############################################################
    ############################################################

    ##########################################
    # forward and create soft label embedding
    ##########################################
    

    ## forward using source domain model and source domain data
    python -u run_exp_modified.py --production cfg/commonvoice/forward/commonvoice_GRU_4layer_mfcc_train_us_30000_forward_us_30000.cfg >log/commonvoice/forward/commonvoice_GRU_4layer_mfcc_train_us_30000_forward_us_30000.log 2>&1


    ## create soft label embedding
    alidir=/nobackup/bx/zhuhan/kaldi/kaldi-trunk-20181127/egs/commonvoice_modified/s5/exp/tri3b_ali_valid_train_all
    forwarddir=/nobackup/bx/zhuhan/pytorch-kaldi/exp/commonvoice/forward/commonvoice_GRU_4layer_mfcc_train_us_30000_forward_us_30000
    
    python -u local/create_soft_label_embedding.py --forward-dir $forwarddir --ali-dir $alidir --temp 5 --nj 5 >> log/commonvoice/forward/create_soft_label_embedding.log 2>&1
    python -u local/create_soft_label_embedding.py --forward-dir $forwarddir --ali-dir $alidir --temp 2 --nj 5 >> log/commonvoice/forward/create_soft_label_embedding.log 2>&1
    python -u local/create_soft_label_embedding.py --forward-dir $forwarddir --ali-dir $alidir --temp 1 --nj 5 >> log/commonvoice/forward/create_soft_label_embedding.log 2>&1


    #######################
    # fine-tune 2000 utt
    #######################

    # australia

    python -u run_exp_modified.py --train "0" cfg/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20.cfg > log/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20.log 2>&1
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20.cfg >> log/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20.log 2>>&1


    # england
    
    python -u run_exp_modified.py --train "0" cfg/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20.cfg> log/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20.cfg >> log/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20.log 2>&1

    # indian
    
    python -u run_exp_modified.py --train "0" cfg/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20.cfg > log/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20.cfg >> log/commonvoice/fine-tune/commonvoice_GRU_4layer_mfcc_train_us_30000_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20.log 2>&1



    #######################
    # soft fine-tune 2000 utt
    #######################

    # australia

    ## temp_1

    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1.cfg > log/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1.cfg >> log/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1.log 2>&1

    ## temp_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2.cfg > log/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2.cfg >> log/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2.log 2>&1

    ## temp_5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5.cfg > log/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5.cfg >> log/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5.log 2>&1


    # england

    ## temp_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1.cfg > log/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1.cfg >> log/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1.log 2>&1


    ## temp_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2.cfg > log/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2.cfg >> log/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2.log 2>&1

    ## temp_5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5.cfg > log/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5.cfg >> log/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5.log 2>&1


    # indian

    ## temp_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1.cfg > log/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1.cfg >> log/commonvoice/soft_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1.log 2>&1

    ## temp_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2.cfg > log/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2.cfg >> log/commonvoice/soft_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2.log 2>&1

    ## temp_5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5.cfg > log/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5.cfg >> log/commonvoice/soft_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5.log 2>&1


    #######################
    # soft hard fine-tune 2000 utt
    #######################

    # australia

    ## temp_1

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1
    
    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1


    ## temp_2

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1
    
    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1


    ## temp_5

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1
    
    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1

    # england


    ## temp_1

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1
    
    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1


    ## temp_2

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1
    
    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1


    ## temp_5

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1
    
    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1



    # indian


    ## temp_1

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1
    
    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1


    ## temp_2

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1
    
    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1


    ## temp_5

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1
    
    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg > log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg >> log/commonvoice/soft_hard_fine-tune/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_soft_hard_fine-tune_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1




    #######################
    # distillation 2000 utt
    #######################

    # australia

    ## temp_1

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1


    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1


    ### weight_2 
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_2.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_2.log 2>&1
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_2.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_1_weight_2.log 2>&1


    ## temp_2

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1

    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_2.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_2.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_2_weight_2.log 2>&1


    ## temp_5

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1

    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_2.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_2.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_20_temp_5_weight_2.log 2>&1


    # england

    ## temp_1

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1


    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1


    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1


    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_2.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_2.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_1_weight_2.log 2>&1


    ## temp_2

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1


    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1


    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_2.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_2.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_2_weight_2.log 2>&1



    ## temp_5

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1


    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1

    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_1.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1




    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_2.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_2.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_20_temp_5_weight_2.log 2>&1

    # indian

    ## temp_1


    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.1.log 2>&1


    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.2.log 2>&1



    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_0.5.log 2>&1


    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_1.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_1.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_1.log 2>&1



    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_2.cfg > log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_2.log
	
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_2.cfg >> log/commonvoice/distillation/temp_1/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_1_weight_2.log 2>&1


    ## temp_2


    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.2.log 2>&1


    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_0.5.log 2>&1


    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_1.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_1.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_1.log 2>&1


    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_2.cfg > log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_2.log
	
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_2.cfg >> log/commonvoice/distillation/temp_2/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_2_weight_2.log 2>&1


    ## temp_5

    ### weight_0.1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.1.log 2>&1

    ### weight_0.2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.2.log 2>&1

    ### weight_0.5
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_0.5.log 2>&1


    ### weight_1
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_1.log
    
    python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_1.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_1.log 2>&1


    ### weight_2
    python -u run_exp_modified.py --train "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_2.cfg > log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_2.log
	python -u run_exp_modified.py --decode "0" cfg/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_2.cfg >> log/commonvoice/distillation/temp_5/commonvoice_GRU_4layer_mfcc_train_us_30000_distillation_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_20_temp_5_weight_2.log 2>&1


    #######################
    # compute frame accuracy
    #######################

    python -u run_exp_modified.py --train "0" cfg/commonvoice/compute_frame_acc/commonvoice_GRU_4layer_mfcc_train_us_30000_compute_frame_acc_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_1.cfg >log/commonvoice/compute_frame_acc/commonvoice_GRU_4layer_mfcc_train_us_30000_compute_frame_acc_train_australia_2000_valid_australia_1000_test_australia_1000_lr_4e-4_epoch_1.log

    python -u run_exp_modified.py --train "0" cfg/commonvoice/compute_frame_acc/commonvoice_GRU_4layer_mfcc_train_us_30000_compute_frame_acc_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_1.cfg >log/commonvoice/compute_frame_acc/commonvoice_GRU_4layer_mfcc_train_us_30000_compute_frame_acc_train_england_2000_valid_england_1000_test_england_1000_lr_4e-4_epoch_1.log

    python -u run_exp_modified.py --train "0" cfg/commonvoice/compute_frame_acc/commonvoice_GRU_4layer_mfcc_train_us_30000_compute_frame_acc_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_1.cfg >log/commonvoice/compute_frame_acc/commonvoice_GRU_4layer_mfcc_train_us_30000_compute_frame_acc_train_indian_2000_valid_indian_1000_test_indian_1000_lr_4e-4_epoch_1.log






