# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --master_port 12345 \
    train.py --outdir=training-runs/urban/denoise \
    --exp_name_suffix="samples_15M" \
    --data_dir=data/urban --train_mode=denoise --init_dir=init_results/urban \
    --sig_data 1.0 \
    --p_mean=-0.5  --p_std=1.5 \
    --lr=2e-4 --batch=104 --batch-gpu=26  \
    --tick=5 --snap=100 --dump=1000 \
    --workers=8 --duration=15 \
    --guide_ckpt="training-runs/urban_pretrained_ckpts/guide/polydiffuse-urban-guide/network-snapshot.pth" \


# Single GPU debugging
#CUDA_VISIBLE_DEVICES=1 python train.py --outdir=training-runs/urban/denoise \
#   --data_dir=data/urban --init_dir=init_results/urban \
#   --train_mode=denoise \
#   --batch=8  --lr=2e-4 --tick=1 --snap=5 \
#   --p_mean=-0.5 --p_std=1.5 --sig_data=1.0 \
#   --guide_ckpt="training-runs/urban_pretrained_ckpts/guide/polydiffuse-urban-guide/network-snapshot.pth" \
#   --exp_name_suffix="debug_single_gpu"
