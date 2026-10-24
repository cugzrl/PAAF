# ImageNet ResNet-50

PAAF initializes the image backbone from the PyTorch ResNet-50 checkpoint `resnet50-19c8e357.pth`.

```bash
cd checkpoints
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

The training config loads this file as `../checkpoints/resnet50-19c8e357.pth` when commands are run from `tools/`.
