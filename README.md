# Dynamic-Neural-Regeneration-DNR

This repository contains the official implementation of the NeurIPS paper **Dynamic Neural Regeneration: Enhancing Deep Learning Generalization on Small Datasets** [[Paper](https://openreview.net/pdf?id=qCpCy0EQAJ)] by **Vijaya Raghavan T Ramkumar, Elahe Arani and Bahram Zonooz**. 

## Abstract
The efficacy of deep learning techniques is contingent upon access to large volumes
of data (labeled or unlabeled). However, in practical domains such as medical
applications, data availability is often limited. This presents a significant challenge:
How can we effectively train deep neural networks on relatively small datasets
while improving generalization? Recent works have explored evolutionary or
iterative training paradigms, which reinitialize a subset of parameters to enhance
generalization performance for small datasets. However, these methods typically
rely on randomly selected parameter subsets and maintain fixed masks throughout
training, potentially leading to suboptimal outcomes. Inspired by neurogenesis in
the brain, we propose a novel iterative training framework, Dynamic Neural Regeneration (DNR), that employs a data-aware dynamic masking scheme to eliminate
redundant connections by estimating their significance. This approach increases
the model’s capacity for further learning through random weight reinitialization.
Experimental results demonstrate that our approach outperforms existing methods
in accuracy and robustness, highlighting its potential for real-world applications
where data collection is challenging
![alt text](https://github.com/NeurAI-Lab/Dynamic-Neural-Regeneration/blob/main/DNR/DNR_method.png) 

For more details, please see the [Paper](https://openreview.net/pdf?id=qCpCy0EQAJ).

## Requirements

The code has been built from the repository of fortuitous forgetting and knowledge evolution paper. To install the required packages: 
```bash
$ pip install -r requirements.txt
```


### Training 

To run training of DNR:

```
$ python .\DNR\train_KE_cls.py  --weight_decay 0.0001 --arch Split_ResNet18 --no_wandb --set FLOWER102 --data /data/input-ai/datasets/flower102 \
           --epochs 200 --num_generations 11  --sparsity 0.8 --save_model --snip
          
```


To run training of LLF: 

```
$ python .\DNR\train_KE_cls.py  --weight_decay 0.0001 --arch Split_ResNet18 --no_wandb --set FLOWER102 --data /data/input-ai/datasets/flower102 \
           --epochs 200 --num_generations 11  --sparsity 0.8 --save_model --reset_layer_name layer4
          
```
To run training of KE:

```
$ python .\DNR\train_KE_cls.py  --weight_decay 0.0001 --arch Split_ResNet18 --no_wandb --set FLOWER102 --data /data/input-ai/datasets/flower102 \
           --epochs 200 --num_generations 11  --sparsity 0.8 --save_model --split_rate 0.8 --split_mode kels
          
```

To run generations-equivalent of the long baseline:


```
$ python .\DNR\train_KE_cls.py  --weight_decay 0.0001 --arch Split_ResNet18 --no_wandb --set FLOWER102 --data /data/input-ai/datasets/flower102 \
           --epochs 2200 --num_generations 1  
          
```

To run layerwise reinitialization:


```
$ python .\DNR\train_KE_cls_LW.py  --weight_decay 0.0001 --arch Split_ResNet18 --no_wandb --set FLOWER102 --data /data/input-ai/datasets/flower102 \
           --epochs 200 --num_generations 8  
          
```



## Reference & Citing this work

If you use this code in your research, please cite the original work [[Paper](https://openreview.net/pdf?id=qCpCy0EQAJ)] :

```
@inproceedings{ramkumardynamic,
  title={Dynamic Neural Regeneration: Enhancing Deep Learning Generalization on Small Datasets},
  author={Ramkumar, Vijaya Raghavan T and Arani, Elahe and Zonooz, Bahram},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}


```


