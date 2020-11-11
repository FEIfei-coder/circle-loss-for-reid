# circle-loss-for-reid
we show simple reproduction of circle loss for reid. 
*we let resnet50 as the baseline without any trick.*

parper: [circle loss](https://arxiv.org/pdf/2002.10857.pdf) 

## But unfortunately, we did not achieve the accuracy rate mentioned in the paper by resnet50 baseline
### we are adjusting the hyperparameters 
| rank-1 | mAP   | gamma | m    | lr     | lrs  | step | batch |
| ------ | ----- | ----- | ---- | ------ | ---- | ---- | ----- |
| 72.1%  | 51.7% | 80    | 0.3  | 0.0003 | 0.1  | 60   | 32    |
| 59.1%  |       | 80    | 0.3  | 0.003  | 0.5  | 20   | 32    |
| 68.4%  |       | 80    | 0.3  | 0.0003 | 0.4  | 60   | 32    |
| 71.9%  | 51.7% | 128   | 0.3  | 0.0003 | 0.5  | 30   | 32    |


# Reference
The codes are expanded on 
- [ReID-baseline](https://github.com/michuanhaohao/deep-person-reid) by Luo & Liao 
