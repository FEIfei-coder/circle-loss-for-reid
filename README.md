# circle-loss-for-reid
we show simple reproduction of circle loss for reid. 
we let resnet50 as the baseline without any trick.

paper: [https://arxiv.org/pdf/2002.10857.pdf](https://arxiv.org/pdf/2002.10857.pdf) 
#### But unfortunately, we did not achieve the accuracy rate mentioned in the paper by resnet50 baseline 
we are adjusting the hyperparameters 

## CircleLoss is in losses.py

# Run
```
>>> python train2.py
```

# Implement
- Dataset: market1501
- baseline: resnet50 without fc and tricks

| rank-1 | mAP   | gamma | m    | lr      | lr-decay  | step | batch |
| ------ | ----- | ----- | ---- | ------- | ---- | ---- | ----- |
| 82%  |       | 56    | 0.3  | 0.00035 | 0.1  | 30   | 128    | 
   

## when the loss reach [4,7], it will fluctuate. 

# circle loss
the dist of the circle loss achieved by cos dist in paper, we add euclidean dist in the loss.py

# Reference
The codes are expanded on 
- [ReID-baseline](https://github.com/michuanhaohao/deep-person-reid) by Luo & Liao 
