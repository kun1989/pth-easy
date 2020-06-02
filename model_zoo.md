# Model Zoo
```
train paramters: 
200 epoch, 5 warmup epoch, 256 batchsize, 0.1 learning rate, 0.0001 weight decay, 
apex automatic mixed precision, label smooth
```
|     network                       | accuracy | Mixup or not | dali or not |
| :-------------------------------: | :------: | :----------: | :---------: |
|  Resnet50 V1[1]                   |  77.948  |     yes      |      not    |
|  Resnet50 V1b[2]                  |  78.378  |     yes      |      not    |
|  Resnext50                        |  79.192  |     yes      |      not    |
|  Mobilenet V1                     |  73.814  |     not      |      not    |
|  Mobilenet V2                     |  72.456  |     not      |      not    |
|  Mobilenet V3                     |  73.890  |     not      |      not    |

[1]: 78.132 when batchsize is 1024
[2]: stride 2 on 3x3 conv

use the new techniques in Mobilenet V3 for Mobilenet V2:
|     network                       | accuracy | Mixup or not | dali or not |
| :-------------------------------: | :------: | :----------: | :---------: |
|  Mobilenet V2                     |  72.393  |     not      |      yes    |
|  Mobilenet V2 relu                |  72.887  |     not      |      yes    |
|  Mobilenet V2 relu 5x5            |  73.700  |     not      |      yes    |
|  Mobilenet V2 relu 5x5 se         |  74.822  |     not      |      yes    |
|  Mobilenet V2 5x5 se hard_swish   |  75.521  |     not      |      yes    |

mixup (not useful for small network)：
|     network                       | use      | not use  | 
| :-------------------------------: | :------: | :------: |
|  Resnet50 V1                      |  77.948  |  77.498  |   
|  Mobilenet V1                     |  73.530  |  73.814  |    
|  Mobilenet V2                     |  71.844  |  72.456  | 

dali (a little drop):
|     network                       | dali     | pil      | 
| :-------------------------------: | :------: | :------: |
|  Resnet50 V1                      |  77.460  |  77.948  |   
|  Mobilenet V2                     |  72.393  |  72.456  | 

