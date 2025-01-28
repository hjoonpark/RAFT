#!/bin/bash
clear
mkdir -p checkpoints
# rm -rf logs
# nohup python -u train.py --name raft-tseries2d --stage tseries2d --validation tseries2d --gpus 0 --num_steps 1200000000 --batch_size 4 --lr 0.0001 --image_size 512 512 --wdecay 0.0001 --mixed_precision > log.nohup &
python -u train.py --name raft-tseries2d --stage tseries2d --validation tseries2d --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.00005 --image_size 512 512 --wdecay 0.0001 --mixed_precision
# nohup python -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 --num_steps 1200000 --batch_size 1 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision > log.nohup &
# python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --mixed_precision
# python -u train.py --name raft-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision
# python -u train.py --name raft-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 --num_steps 50000 --batch_size 5 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision
