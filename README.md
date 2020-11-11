# circle-loss-for-reid
we show simple reproduction of circle loss for reid. 
we let resnet50 as the baseline without any trick.

paper: [https://arxiv.org/pdf/2002.10857.pdf](https://arxiv.org/pdf/2002.10857.pdf) 
#### But unfortunately, we did not achieve the accuracy rate mentioned in the paper by resnet50 baseline 
we are adjusting the hyperparameters 

## problem above analysis
we regard circle loss as a embedding loss, so we use randomsampler function when we load the data.

# Run
```
>>> python train2.py
```

# Implement (use randomsampler func)
- Dataset: market1501
- baseline: resnet50 without fc and tricks

| rank-1 | mAP   | gamma | m    | lr     | lrs  | step | batch |        
| ------ | ----- | ----- | ---- | ------ | ---- | ---- | ----- | 
| 72.4%  | 51.7% | 80    | 0.3  | 0.0003 | 0.1  | 60   | 32    |        
| 59.1%  |       | 80    | 0.3  | 0.003  | 0.5  | 20   | 32    | 
| 68.4%  |       | 80    | 0.3  | 0.0003 | 0.4  | 60   | 32    | 
| 69.8%  |       | 80    | 0.2  | 0.0003 | 0.1  | 60   | 32    |        
|        |       | 80    | 0.25 | 0.0006 | 0.1  | 90   | 32    |        
| 71.9%  | 51.7% | 128   | 0.3  | 0.0003 | 0.5  | 30   | 32    | 
| 68.4%  | 45.8% | 128   | 0.25 | 0.0003 | 0.1  | 60   | 64    |       
| 69.4%  | 43.9% | 128   | 0.3  | 0.0003 | 0.5  | 60   | 64    |        
| 71.9%  | 51.9% | 128   | 0.25 | 0.0003 | 0.1  | 60   | 32    |        
| 63.6%  |       | 128   | 0.25 | 0.0003 | 0.1  | 30   | 64    |        
|        |       | 128   | 0.2  | 0.0005 | 0.1  | 60   | 32    |        
| 70.3%  |       | 256   | 0.2  | 0.0003 | 0.1  | 60   | 32    |               



# circle loss
the dist of the circle loss achieved by cos dist in paper, we add euclidean dist in the loss.py

# Reference
The codes are expanded on 
- [ReID-baseline](https://github.com/michuanhaohao/deep-person-reid) by Luo & Liao 
