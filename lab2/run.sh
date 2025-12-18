#!/usr/bin/env bash
set -e

########################
# 基本配置
########################
GPU_ID=0
SEED=2024

DATA_DIR=/root/autodl-tmp/lab2/data/
SAVE_DIR=checkpoints/kg_transe

########################
# 模型配置
########################
KG_TYPE=TransE         # TransE | TransR
EMBED_DIM=200
RELATION_DIM=200       # TransR 时可以改成 100

########################
# 训练配置
########################
EPOCH=500
BATCH_SIZE=4096
LR=0.0005
L2_LAMBDA=1e-5

EVAL_EVERY=5
PRINT_EVERY=50
STOPPING_STEPS=10

########################
# 评估配置
########################
KS="(1,3,10)"
TEST_BATCH_SIZE=128

########################
# 启动
########################
python main_kg.py \
  --cuda 1 \
  --gpu_id ${GPU_ID} \
  --seed ${SEED} \
  --data_dir ${DATA_DIR} \
  --KG_embedding_type ${KG_TYPE} \
  --embed_dim ${EMBED_DIM} \
  --relation_dim ${RELATION_DIM} \
  --n_epoch ${EPOCH} \
  --kg_batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --kg_l2loss_lambda ${L2_LAMBDA} \
  --evaluate_every ${EVAL_EVERY} \
  --print_every ${PRINT_EVERY} \
  --stopping_steps ${STOPPING_STEPS} \
  --Ks "${KS}" \
  --test_batch_size ${TEST_BATCH_SIZE}
