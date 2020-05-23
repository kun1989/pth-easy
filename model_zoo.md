# Model Zoo
```
train paramters: 
200 epoch, 5 warmup epoch, 256 batchsize, 0.1 learning rate, 0.0001 weight decay, 
apex automatic mixed precision, label smooth
```
|     network     | accuracy | Mixup or not |
| :-------------: | :------: |   :------:   |
|  Resnet50 V1    |  77.948  |     yes      |
|  Mobilenet V1   |  73.814  |     not      |
|  Mobilenet V2   |  72.456  |     not      |
|  Mobilenet V3   |  73.890  |     not      |
