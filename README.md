
## DualRec (Disentangling Past-Future Modeling in Sequential Recommendation via Dual Networks)    
Mindspore Version code of CIKM 2022 Paper "[Disentangling Past-Future Modeling in Sequential Recommendation via Dual Networks](https://dl.acm.org/doi/10.1145/3511808.3557289)"

## Description   
Sequential recommendation (SR) plays an important role in personalized recommender systems because it captures dynamic and diverse preferences from users' real-time increasing behaviors. Unlike the standard autoregressive training strategy, future data (also available during training) has been used to facilitate model training as it provides richer signals about users' current interests and can be used to improve the recommendation quality. However, existing methods suffer from a severe training-inference gap, i.e., both past and future contexts are modeled by the same encoder when training, while only historical behaviors are available during inference. This discrepancy leads to potential performance degradation. To alleviate the training-inference gap, we propose a new framework DualRec, which achieves past-future disentanglement and past-future mutual enhancement by a novel dual network. Specifically, a dual network structure is exploited to model the past and future context separately.And a bi-directional knowledge transferring mechanism enhances the knowledge learnt by the dual network. Extensive experiments on four real-world datasets demonstrate the superiority of our approach over baseline methods. Besides, we demonstrate the compatibility of DualRec by instantiating using different backbones. Further empirical analysis verifies the high utility of modeling future contexts under our DualRec framework.

## Dependencies
- mindspore-gpu == 1.9.0
- numpy
- pandas
- tqdm
- pyyaml

## Datasets
The dataset is available in the ./Data folder.
- [Beauty](http://jmcauley.ucsd.edu/data/amazon/)
- [Sports](http://jmcauley.ucsd.edu/data/amazon/)
- [Toys](http://jmcauley.ucsd.edu/data/amazon/)
- [Yelp](https://www.yelp.com/dataset/)

## How to run   
First, install dependencies   
```bash
# install dependencies   
cd DualRec
pip install -r requirements.txt
```
 Next, run the model with config files for corresponding datasets
 ```bash
python run.py --config src/config/{datasets_name}.yaml
 ```

## Cite our work
```
@inproceedings{10.1145/3511808.3557289,
author = {Zhang, Hengyu and Yuan, Enming and Guo, Wei and He, Zhicheng and Qin, Jiarui and Guo, Huifeng and Chen, Bo and Li, Xiu and Tang, Ruiming},
title = {Disentangling Past-Future Modeling in Sequential Recommendation via Dual Networks},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557289},
doi = {10.1145/3511808.3557289},
abstract = {Sequential recommendation (SR) plays an important role in personalized recommender systems because it captures dynamic and diverse preferences from users' real-time increasing behaviors. Unlike the standard autoregressive training strategy, future data (also available during training) has been used to facilitate model training as it provides richer signals about users' current interests and can be used to improve the recommendation quality. However, existing methods suffer from a severe training-inference gap, i.e., both past and future contexts are modeled by the same encoder when training, while only historical behaviors are available during inference. This discrepancy leads to potential performance degradation. To alleviate the training-inference gap, we propose a new framework DualRec, which achieves past-future disentanglement and past-future mutual enhancement by a novel dual network. Specifically, a dual network structure is exploited to model the past and future context separately.And a bi-directional knowledge transferring mechanism enhances the knowledge learnt by the dual network. Extensive experiments on four real-world datasets demonstrate the superiority of our approach over baseline methods. Besides, we demonstrate the compatibility of DualRec by instantiating using different backbones. Further empirical analysis verifies the high utility of modeling future contexts under our DualRec framework.},
booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
pages = {2549–2558},
numpages = {10},
keywords = {dual network, training-inference gap, sequential recommendation},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```
