#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 train.py --launcher pytorch > log.txt&
