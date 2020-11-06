import os
from utils import run_shell
from data_io import read_vec_int_ark

def create_clean_real_simu_dir(egs_dir, clean_dir_list, real_dir_list, simu_dir_list):
    total_clean_dir = os.path.join(egs_dir,"data", "total_clean")
    total_real_dir = os.path.join(egs_dir,"data", "total_real")
    total_simu_dir = os.path.join(egs_dir,"data", "total_simu")
    for dir in [total_clean_dir, total_real_dir, total_simu_dir]:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    for (dir_list, total_dir) in [(clean_dir_list, total_clean_dir), (real_dir_list, total_real_dir), (simu_dir_list, total_simu_dir)]:
        total_file = []
        for dir in dir_list:
            with open(os.path.join(egs_dir,"data", dir, "feats.lengths"), "r") as f_in:
                single_file = f_in.readlines()
                total_file.extend(single_file)
        with open(os.path.join(egs_dir, "data", total_dir, "feats.lengths"), "w") as f_out:
            f_out.write("".join(total_file))


def create_label(feats_file_list, label_file, ref_label_folder_list, log_dir):
    
    label_dict = {}
    ref_lab_dict = {}
    label_string = ""
    for index in range(0, len(feats_file_list)):
        with open(feats_file_list[index], "r") as f_in:
            for line in f_in.readlines():
                sentence_index = line.split()[0]
                sentence_len = line.split()[1]
                label_dict[sentence_index] = (index, sentence_len)
    for ref_label_folder in ref_label_folder_list:
        ref_lab_dict_part = {
            k: v
            for k, v in read_vec_int_ark(
                "gunzip -c " + ref_label_folder + "/ali*.gz | " + "ali-to-pdf " + " " + ref_label_folder + "/final.mdl ark:- ark:-|",
                log_dir,
            )
        }
        ref_lab_dict.update(ref_lab_dict_part)
    
    print("before: ", len(label_dict))
    label_dict = {
        k: v for k, v in label_dict.items() if k in ref_lab_dict
    } 
    print("after: ", len(label_dict))

    for sentence_index, (index, sentence_len) in label_dict.items():
        label_string += sentence_index + " "
        label_string += (str(index) + " ") * (int(sentence_len) -1)
        label_string += str(index) + "\n"
    with open(label_file, "w") as f_out:
        f_out.write(label_string)

if __name__ == "__main__":
    # combine orig_clean, real_noisy and simu_noisy feats.length
    clean_dir_list = "tr05_orig_clean dt05_orig_clean et05_orig_clean".split()
    real_dir_list = "tr05_real_noisy dt05_real_noisy et05_real_noisy".split()
    simu_dir_list = "tr05_simu_noisy dt05_simu_noisy et05_simu_noisy".split()
    egs_dir = "/home/zhuhan/NasStore/kaldi-trunk/egs/chime3_modified/s5"
    create_clean_real_simu_dir(egs_dir, clean_dir_list, real_dir_list, simu_dir_list)

    # create label
    feats_clean = os.path.join(egs_dir,"data", "total_clean", "feats.lengths")
    feats_real = os.path.join(egs_dir,"data", "total_real", "feats.lengths")
    feats_simu = os.path.join(egs_dir,"data", "total_simu", "feats.lengths")
    feats_file_list = [feats_clean, feats_real, feats_simu]
    label_dir = "/home/zhuhan/NasStore/kaldi-trunk/egs/chime3_modified/s5/exp/lab_domain_classify"
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    label_file = os.path.join(label_dir, "ali1.txt")
    ref_label_folder_list = ["/home/zhuhan/NasStore/kaldi-trunk/egs/chime3_modified/s5/exp/tri3b_ali_tr05_multi_noisy_clean", 
    "/home/zhuhan/NasStore/kaldi-trunk/egs/chime3_modified/s5/exp/tri3b_ali_tr05_orig_clean_dt05_orig_clean", 
    "/home/zhuhan/NasStore/kaldi-trunk/egs/chime3_modified/s5/exp/tri3b_converted_ali_tr05_multi_noisy_dt05_multi_noisy", 
    "/home/zhuhan/NasStore/kaldi-trunk/egs/chime3_modified/s5/exp/tri3b_ali_tr05_orig_clean_et05_orig_clean", 
    "/home/zhuhan/NasStore/kaldi-trunk/egs/chime3_modified/s5/exp/tri3b_converted_ali_tr05_multi_noisy_et05_multi_noisy"]
    create_label(feats_file_list, label_file, ref_label_folder_list, label_dir)

    # transform label to binary
    ali_file = os.path.join(label_dir, "ali1")
    
    log_file = os.path.join(label_dir, "ali.log")
    cmd = "copy-int-vector --binary=true ark:" + label_file + " ark:" + ali_file
    run_shell(cmd, log_file)

    # zip the label
    cmd = "gzip " + ali_file
    run_shell(cmd, log_file)
    
    print("domain classify label creation success, lab is stored as {}.".format(ali_file + ".gz"))
