# TUCH
> Source code of paper "Transformer-based Unsupervised Cross-modal Hashing with Weight Fusion"
## Introduction
With the explosive growth of information on the Internet, cross-modal retrieval has become an important and valuable frontier hot-spot. Deep hashing has achieved great success in cross-modal retrieval due to its low storage consumption and high search speed. However, most deep cross-modal hashing methods construct parallel networks to process multi-modal data, which ignores the integrated representation in view of the cross-modal graphic information. In this paper, we propose a novel unsupervised cross-modal hashing method by constructing two mode-specific encoders and a fusion module. The fusion module is designed to associate various modes to mine the semantic structure of the cross-modal data. The joint consistent loss is constructed to preserve inter-modal and intra-modal similarities simultaneously. In addition, we utilize Swin Transformer backbone to extract more discriminative image embeddings instead of commonly used convolutional neural networks. Experiments on three cross-modal datasets show that the proposed method obtains superior accuracy in comparison with state-of-the-art cross-modal hashing baselines.
## Requirements
- Python: 3.x
- h5py: 3.7.0
- numpy: 1.21.6
- pytorch: 1.12.0
- scipy: 1.7.3
## Run
- Update the [settings.py](https://github.com/sellaner/TUCH/blob/main/source/settings.py) with your `data_dir`. And change the value **'EVAL'**, for **train** setting it with `False`
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
- [DJSRH](https://github.com/zs-zhong/DJSRH)
