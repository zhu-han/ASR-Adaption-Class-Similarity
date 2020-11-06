import glob
import numpy as np
import torch
import pickle
import os
from data_io import read_mat_ark, read_vec_int_ark
from utils import run_shell
import time
import copy
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Creat soft label embedding/vector')

parser.add_argument('--label-type', type=str, default="embedding", \
    help="create embedding or vector.")

parser.add_argument('--temp', type=int, default=1, \
    help="Temparature of softmax.")
parser.add_argument('--nj', type=int, default=8, \
    help="max processing number.")
parser.add_argument('--forward-dir', type=str, default= "/NasStore/5/speech-students/zhuhan/pytorch-kaldi/exp/commonvoice/forward/commonvoice_GRU_4layer_mfcc_train_us_30000_forward_us_30000", \
    help="dir where prediction of neural net stored")
parser.add_argument('--ali-dir', type=str, default= "/NasStore/5/speech-students/zhuhan/kaldi-trunk/egs/commonvoice_modified/s5/exp/tri3b_ali_valid_train_us_30000", \
    help="dir where alignments stored")
args = parser.parse_args()



def soft_label_embedding_creation(forward_dir, ali_dir, temp, max_processing_num):
    log_dir = forward_dir
    cmd = "hmm-info " + ali_dir + "/final.mdl | awk '/pdfs/{print $4}'"
    pdf_num = run_shell(cmd, log_dir+"/log.log")
    manager = multiprocessing.Manager()

    embedding_list = manager.list()
    total_pdf_list = manager.list()
    prediction_ark_list = sorted(glob.glob(forward_dir + "/exp_files/" +"*ark"))

    # read lab from ali
    lab = {
    k: v
    for k, v in read_vec_int_ark(
        "gunzip -c " + ali_dir + "/ali*.gz | " + "ali-to-pdf" + " " + ali_dir + "/final.mdl ark:- ark:-|",
        log_dir)
    }
    processes = []
    # produce soft label embedding and total pdf num in multiprocess
    for process_index in range(0, len(prediction_ark_list)):
        p = multiprocessing.Process(
            target=sub_process,
            kwargs={
                "process_index": process_index,
                "pdf_num": pdf_num,
                "temp": temp,
                "prediction_ark_list": prediction_ark_list,
                "embedding_list": embedding_list,
                "total_pdf_list": total_pdf_list,
                "lab": lab,
                "log_dir": log_dir,
            },
        )
        processes.append(p)
        if len(processes) > max_processing_num:
            processes[0].join()
            del processes[0]
        p.start()
        # sub_process(process_index, prediction_ark_list, embedding_list, total_pdf_list, lab, log_dir)
    for process in processes:
        process.join()
    # sum all the results of multiprocess
    embedding = np.zeros((int(pdf_num), int(pdf_num)), dtype=np.float32)
    total_pdf = np.zeros(int(pdf_num), dtype=np.int64)
    for process_index in range(0, len(prediction_ark_list)):
        embedding += embedding_list[process_index]
        total_pdf += total_pdf_list[process_index]
    for pdf_index in range(0, int(pdf_num)):
        embedding[pdf_index] /= float(total_pdf[pdf_index])
    with open(os.path.join(forward_dir, "temp_" + str(temp) + "_soft_label_embedding.pkl"), "wb") as f_out:
        pickle.dump(embedding, f_out)
    return embedding, total_pdf




def soft_label_vecotor_creation_with_embedding(forward_dir, ali_dir, temp):
    log_dir = forward_dir
    cmd = "hmm-info " + ali_dir + "/final.mdl | awk '/pdfs/{print $4}'"
    pdf_num = run_shell(cmd, log_dir+"/log.log")

    # read lab from ali
    lab = {
    k: v
    for k, v in read_vec_int_ark(
        "gunzip -c " + ali_dir + "/ali*.gz | " + "ali-to-pdf" + " " + ali_dir + "/final.mdl ark:- ark:-|",
        log_dir)
    }
    total_pdf = np.zeros(int(pdf_num), dtype=np.int64)
    for key in lab:
        for pdf_index in range(0,len(lab[key])):
            # sum over all frames of all sample
            total_pdf[lab[key][pdf_index]] += 1

    # sum all the results of multiprocess
    embedding = np.zeros((int(pdf_num), int(pdf_num)), dtype=np.float32)

    with open(os.path.join(forward_dir, "temp_" + str(temp) + "_soft_label_embedding.pkl"), "rb") as f_in:
        embedding = pickle.load(f_in)

    for pdf_index in range(0, int(pdf_num)):
        embedding[pdf_index] *= float(total_pdf[pdf_index])
    vector = embedding.sum(axis=0)
    vector = vector / total_pdf.sum()

    with open(os.path.join(forward_dir, "temp_" + str(temp) + "_soft_label_vector.pkl"), "wb") as f_out:
        pickle.dump(vector, f_out)
    return vector, total_pdf

def sub_process(process_index, pdf_num, temp,  prediction_ark_list, embedding_list, total_pdf_list, lab, log_dir):
    print("start processing the {}-th ark".format(process_index))
    start_time = time.time()

    # read prediction of neural net
    read_prediction = {
            k: m
            for k, m in read_mat_ark("ark:" + prediction_ark_list[process_index], log_dir)
            if k in lab
        }
    prediction = copy.deepcopy(read_prediction)
    del read_prediction
    for key in prediction:
        for pdf_index in range(0,len(prediction[key])):
            # compute high temp softmax of all frames of all sample
            prediction[key][pdf_index] = prediction[key][pdf_index] / temp
            prediction[key][pdf_index] = np.exp(prediction[key][pdf_index] - np.max(prediction[key][pdf_index]))/sum(np.exp(prediction[key][pdf_index] - np.max(prediction[key][pdf_index])))

    embedding = np.zeros((int(pdf_num), int(pdf_num)), dtype=np.float32)
    total_pdf = np.zeros(int(pdf_num), dtype=np.int64)
    for key in prediction:
        for pdf_index in range(0,len(prediction[key])):
            # sum over all frames of all sample
            embedding[lab[key][pdf_index]] += prediction[key][pdf_index]
            total_pdf[lab[key][pdf_index]] += 1
    del prediction
    embedding_list.append(embedding)
    total_pdf_list.append(total_pdf)
    end_time = time.time()
    print('finished the {}-th ark, using time: {}s ==='.format(process_index, round(end_time - start_time)))


if __name__ == "__main__":
    label_type = args.label_type
    forward_dir = args.forward_dir
    ali_dir = args.ali_dir
    temp = args.temp
    max_processing_num = args.nj
    print("start create soft label {} of temp {}".format(label_type, temp))
    total_start_time = time.time()
    if label_type == "embedding":
        embedding, total_pdf = soft_label_embedding_creation(forward_dir, ali_dir, temp, max_processing_num)
    elif label_type == "vector":
        vector, total_pdf = soft_label_vecotor_creation_with_embedding(forward_dir, ali_dir, temp)
    total_end_time = time.time()
    print("soft label {} of temp {} creation success, using time: {}s, the embedding file is stored in {}".format(label_type, temp, round(total_end_time - total_start_time), forward_dir))
