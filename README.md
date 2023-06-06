# UECA-Prompt: Universal Prompt for Emotion Cause Analysis

## Prerequisites

Python 3.8  
[Pytorch](https://pytorch.org/) 1.9.0  
[CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) 10.1  
BERT - our BERT model is adapted from this implementation:  
https://github.com/huggingface/pytorch-pretrained-BERT  

## Dataset

- ```divide_fold.py```: used to  get 20 files, which will be named as “foldx_train.txt” and “foldx_test.txt”, where “x” should be from 1 to 10.  

**data_combine_ECPE** - A dir where contains data splits for ECPE task. The test dataset are named as fold\*\_test.txt, while the train datasets are named as fold\*\_train.txt.

**data_combine_ECE** - A dir where contains data splits for ECE task. The test dataset are named as fold\*\_test.txt, while the train datasets are named as fold\*\_train.txt.

**data_combine_CCRC** - A dir where contains data splits for CCRC task. The test dataset are named as fold\*\_test.txt, while the train datasets are named as fold\*\_train.txt.

- ```preprocess.py```: used to get the manually labeled datase.  

- ```gen_nega_samples.py```: used to  generate the constructed conditional-ECPE dataset.  

**data_combine_ECE_balance** - A dir where contains data splits for de-bias dataset for ECE task. The test dataset are named as fold\*\_test.txt, while the train datasets are named as fold\*\_train.txt.

**data_combine_ECPE_balance** - A dir where contains data splits for de-bias dataset for ECPE task. The test dataset are named as fold\*\_test.txt, while the train datasets are named as fold\*\_train.txt.

## Usage

Download checkpoint from https://www.dropbox.com/sh/45jj8dcenhbuzvn/AABbXSxccgyi1AMGA5yi4DBUa?dl=0 and save in the fold ```checkpoint```

Download pretraind model  from https://huggingface.co/bert-base-chinese and save it as ```bert-base-chinese```.

- run ```ECE.py``` for ECE task.

- run ```ECPE.py``` for ECPE task.

- run ```CCRC.py``` for CCRC task.

- run ```ECPE_M2M.py``` for  M2M variant method in ECPE task.


## Citation
If you find our work useful, please consider citing UECA-Prompt:

```
@inproceedings{zheng2022ueca,
  title={UECA-Prompt: Universal Prompt for Emotion Cause Analysis},
  author={Zheng, Xiaopeng and Liu, Zhiyue and Zhang, Zizhen and Wang, Zhaoyang and Wang, Jiahai},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={7031--7041},
  year={2022}
}
```
[![Page Views Count](https://badges.toozhao.com/badges/01H17FAJZN689KVAB00C5XR50P/green.svg)](https://badges.toozhao.com/stats/01H17FAJZN689KVAB00C5XR50P "Get your own page views count badge on badges.toozhao.com")
