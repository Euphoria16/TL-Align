
# Token-Label Alignment for Vision Transformers

This is the pytorch implementation for the paper: *Token-Label Alignment for Vision Transformers*. This repo contains the implementation of training DeiTs on ImageNet using our token-label alignment. The proposed method can improve the top-1 accuracy on ImageNet by 1.0%, 0.8% and 0.5% for DeiT-tiny, DeiT-small and DeiT-base respectively.


# Usage

## Prerequisites

This repository is built upon the [Timm](https://github.com/rwightman/pytorch-image-models) library and 
the [DeiT](https://github.com/facebookresearch/deit) repository. 

You need to install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Training by token-label aligment
To enable token-label alignment during training, you can simply add a ```--tl-align``` in your training script. For example, for DeiT-small, run:

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_tla.py \
--model deit_small_patch16_224 \
--batch-size 128    \
--mixup 0.0 \
--tl-align \
--data-path /path/to/imagenet  \
--output_dir /path/to/output  \
```
or 

```
bash train_deit_small_tla.sh
```

This should give 80.6% top-1 accuracy after 300 epochs training.

## Evaluation


The evaluation of models trained by our token-label alignment is the same as [timm](https://github.com/rwightman/pytorch-image-models#train-validation-inference-scripts).
You can also find your validation accuracy during training.

For Deit-small, run:
```
python main_tla.py --eval --resume checkpoint.pth --model deit_small_patch16_224 --data-path /path/to/imagenet
```