############################ 启动参数
MASTER_PORT=27757
MASTER_ADDR="127.0.0.1"
NODE_RANK=0
NNODES=1
GPUS_PER_NODE=2
GPUS="0,1"       
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"    

############################ Training Args
OPTS=""
OPTS=$OPTS" --load-dir none"
OPTS=$OPTS" --ckpt-path none"
OPTS=$OPTS" --save-dir experiment/pretrain_default"
OPTS=$OPTS" --config-path config/default_attentio _config.json"

OPTS=$OPTS" --data-root data"
OPTS=$OPTS" --dataset-dir pretrain"
OPTS=$OPTS" --batch-size 64"
OPTS=$OPTS" --clip-grad 1"
OPTS=$OPTS" --seed 3407"
OPTS=$OPTS" --gradient-accumulation 1"

OPTS=$OPTS" --epochs 20"
OPTS=$OPTS" --lr 1e-4"
OPTS=$OPTS" --weight-decay 0"
OPTS=$OPTS" --warmup-iters 3000"

# OPTS=$OPTS" --debugging"

echo ${OPTS}
CUDA_VISIBLE_DEVICES=${GPUS} torchrun ${DISTRIBUTED_ARGS} run_task.py ${OPTS}