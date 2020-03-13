# MadNet
An implementation of the MadNet based on ResNet

## Requirements
- tensorflow == 1.14.0
- keras == 2.2.4


## Training

```bash
python train_MadNet.py -log_path ./MadNet -resnet_depth 56 -dataset mnist -reduce_jacobian_loss -reduce_variance -label_smoothing 0.8
```
