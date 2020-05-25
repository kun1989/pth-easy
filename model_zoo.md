# Model Zoo
```
train paramters: 
200 epoch, 5 warmup epoch, 256 batchsize, 0.1 learning rate, 0.0001 weight decay, 
apex automatic mixed precision, label smooth
```
|     network                       | accuracy | Mixup or not | dali or not |
| :-------------------------------: | :------: | :----------: | :---------: |
|  Resnet50 V1                      |  77.948  |     yes      |      not    |
|  Mobilenet V1                     |  73.814  |     not      |      not    |
|  Mobilenet V2                     |  72.456  |     not      |      not    |
|  Mobilenet V2                     |  72.393  |     not      |      yes    |
|  Mobilenet V2 relu                |  72.887  |     not      |      yes    |
|  Mobilenet V2 relu 5x5            |  73.700  |     not      |      yes    |
|  Mobilenet V2 relu 5x5 se         |  74.822  |     not      |      yes    |
|  Mobilenet V2 5x5 se hard_swish   |  75.521  |     not      |      yes    |
|  Mobilenet V3                     |  73.890  |     not      |      not    |
