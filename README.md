
# Token-Label Alignment for Vision Transformers

This is the pytorch implementation for the paper: *Token-Label Alignment for Vision Transformers*. 

> [Han Xiao\*](https://scholar.google.com/citations?user=N-u2i-QAAAAJ&hl=en), [Wenzhao Zheng\*](https://scholar.google.com/citations?user=LdK9scgAAAAJ&hl=en), [Zheng Zhu](http://www.zhengzhu.net/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), and [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)

![overview](assets/overview.png)

## Highlights

- Improve your ViTs by ~0.7% with a simple --tl-align command.
- Improvements in accuracy, generalization, and robustness **without** additional computation during inference.
- Efficient alignment of token labels without distillation.

## Results

| Model     | Image Size | Params | FLOPs | Top-1 Acc.(\%) | Top-5 Acc.(\%) |
| --------- | ---------- | ------ | ----- | -------------- | -------------- |
| DeiT-T    | $224^2$    | 5.7M   | 1.6G  | 72.2           | 91.3           |
| +TL-Align | $224^2$    | 5.7M   | 1.6G  | **73.2**       | **91.7**       |
| DeiT-S    | $224^2$    | 22M    | 4.6G  | 79.8           | 95.0           |
| +TL-Align | $224^2$    | 22M    | 4.6G  | **80.6**       | 95.0           |
| DeiT-B    | $224^2$    | 86M    | 17.5G | 81.8           | 95.5           |
| +TL-Align | $224^2$    | 86M    | 17.5G | **82.3**       | **95.8**       |
| Swin-T    | $224^2$    | 29M    | 4.5G  | 81.2           | 95.5           |
| +TL-Align | $224^2$    | 29M    | 4.5G  | **81.4**       | **95.7**       |
| Swin-S    | $224^2$    | 50M    | 8.8G  | 83.0           | 96.3           |
| +TL-Align | $224^2$    | 50M    | 8.8G  | **83.4**       | **96.5**       |
| Swin-B    | $224^2$    | 88M    | 15.4G | 83.5           | 96.4           |
| +TL-Align | $224^2$    | 88M    | 15.4G | **83.7**       | **96.5**       |

## Usage

### Prerequisites

This repository is built upon the [Timm](https://github.com/rwightman/pytorch-image-models) library and the [DeiT](https://github.com/facebookresearch/deit) repository. 

You need to install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data are expected to be in the `train/` folder and `val` folder respectively:

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


### Training by token-label alignment
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

This should give 80.6% top-1 accuracy after 300 epochs of training.

### Evaluation

The evaluation of models trained by our token-label alignment is the same as [timm](https://github.com/rwightman/pytorch-image-models#train-validation-inference-scripts).
You can also find your validation accuracy during training.

For Deit-small, run:
```
python main_tla.py --eval --resume checkpoint.pth --model deit_small_patch16_224 --data-path /path/to/imagenet
```



## Citation

If you find this project useful in your research, please cite:

````
@article{xiao2022token,
    title={Token-Label Alignment for Vision Transformers},
    author={Xiao, Han and Zheng, Wenzhao and Zhu, Zheng and Zhou, Jie and Lu, Jiwen},
    year={2022}
}
````