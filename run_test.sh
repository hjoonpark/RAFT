#!/bin/bash
clear
mkdir -p checkpoints
# rm -rf logs
python -u test.py --name raft-tseries2d --stage tseries2d --validation tseries2d --gpus 0 --num_steps 1 --batch_size 1 --lr 0.0001 --image_size 512 512 --wdecay 0.0001 --mixed_precision --restore_ckpt "logs/tseries2d/raft-tseries2d3.pth"
