# MadNet
An implementation of the MadNet based on ResNet

## Training

```bash
python train_MadNet.py -log_path ./MadNet -resnet_depth 56 -dataset mnist -reduce_jacobian_loss -reduce_variance -label_smoothing 0.8
```
