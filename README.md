# TUCH
> Source code of paper "Transformer-based Unsupervised Cross-modal Hashing with Weight Fusion"
## Introduction

## Requirements
- Python: 3.x
- h5py: 3.7.0
- numpy: 1.21.6
- pytorch: 1.12.0
- scipy: 1.7.3
## Run
- Update the [settings.py](https://github.com/idejie/DSAH/blob/master/settings.py) with your `data_dir`. And change the value [` EVAL`](https://github.com/idejie/DSAH/blob/be1f3edba30015b164bc41994067a71273cbeb30/settings.py#L6), for **train** setting it with `False`
- run the `train.py`
  ```shell
  python train.py
  ```
 
## Datasets
For datasets, we follow [Deep Cross-Modal Hashing's Github (Jiang, CVPR 2017)](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_matlab/DCMH_matlab). You can download these datasets from:

- Wikipedia articles, [[Link](http://www.svcl.ucsd.edu/projects/crossmodal/)]

- MIRFLICKR25K, [[OneDrive]](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EpLD8yNN2lhIpBgQ7Kl8LKABzM68icvJJahchO7pYNPV1g?e=IYoeqn), [[Baidu Pan](https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew), password: 8dub]

- NUS-WIDE (top-10 concept), [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EoPpgpDlPR1OqK-ywrrYiN0By6fdnBvY4YoyaBV5i5IvFQ?e=kja8Kj)], [[Baidu Pan](https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ), password: ml4y]



## References
- [DJSRH](https://github.com/zs-zhong/DJSRH))
