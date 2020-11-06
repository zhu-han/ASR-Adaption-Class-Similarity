## Domain Adaptation Using Class Similarity for Robust Speech Recognition
### Han Zhu, Jiangjiang Zhao, Yuling Ren, Li Wang, Pengyuan Zhang

This is the implementation of our paper accepted in Interspeech 2020 and the paper can be downloaded [here](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/3087.pdf).

## Requirement
- The data preparation and GMM-HMM model training require [Kaldi](https://github.com/kaldi-asr/kaldi).
- The NN acoustic model training requires [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi).

## Dataset

#### CommonVoice
- The CommonVoice dataset could be download from [here](https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz).
- The train/dev/test data we used in this work could be found in the `dataset/commonvoice/experiment_csv` directory.
- The data prepare and gmm-hmm training could be done using `kaldi/commonvoice/run.sh`. 
#### CHIME3
- We use the standard train/dev/test data split of CHIME3 dataset.
- The data prepare and gmm-hmm training could be done using `kaldi/chime3/run.sh`. 

## Experiment
- All experiments in the paper cound be conducted using `pytorch-kaldi/run.sh`.
- Configurations are stored in `pytorch-kaldi/cfg`.

## Citation
```
@inproceedings{Zhu2020,
  author={Han Zhu and Jiangjiang Zhao and Yuling Ren and Li Wang and Pengyuan Zhang},
  title={{Domain Adaptation Using Class Similarity for Robust Speech Recognition}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={4367--4371},
  doi={10.21437/Interspeech.2020-3087},
  url={http://dx.doi.org/10.21437/Interspeech.2020-3087}
}
```

## Contact
- zhuhan@hccl.ioa.ac.cn
