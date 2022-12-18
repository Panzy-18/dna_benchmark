############################ 启动参数
MASTER_PORT=27760
MASTER_ADDR="127.0.0.1"
NODE_RANK=0
NNODES=1
GPUS_PER_NODE=1
GPUS="0"       
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"    

############################ Training Args

OPTS=""
OPTS=$OPTS" --load-dir pretrained/track7878"
OPTS=$OPTS" --save-dir experiment/eqtl49_on_track7878"
OPTS=$OPTS" --data-root data"
OPTS=$OPTS" --dataset-dir eqtl49"
OPTS=$OPTS" --batch-size 64"
OPTS=$OPTS" --no-log"

CUDA_VISIBLE_DEVICES=${GPUS} torchrun ${DISTRIBUTED_ARGS} run_eqtl.py ${OPTS}