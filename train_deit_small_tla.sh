
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_tla.py \
--model deit_small_patch16_224 \
--batch-size 128    \
--tl-align \
--data-path /path/to/imagenet  \
--output_dir /path/to/output  \