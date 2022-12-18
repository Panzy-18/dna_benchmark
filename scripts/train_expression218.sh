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
FEATURE_DIR="expression218_feature"

OPTS1=""
OPTS1=$OPTS1" --load-dir pretrained/track7878"
OPTS1=$OPTS1" --save-dir data/${FEATURE_DIR}"
OPTS1=$OPTS1" --data-root data"
OPTS1=$OPTS1" --dataset-dir expression218"
OPTS1=$OPTS1" --batch-size 1"
OPTS1=$OPTS1" --no-log"

OPTS2=""
OPTS2=$OPTS2" --load-dir none"
OPTS2=$OPTS2" --ckpt-path none"
OPTS2=$OPTS2" --save-dir experiment/expression218"
OPTS2=$OPTS2" --config-name config/default_expression_config.json"
OPTS2=$OPTS2" --data-root data"
OPTS2=$OPTS2" --dataset-dir ${FEATURE_DIR}"
OPTS2=$OPTS2" --batch-size 64"
OPTS2=$OPTS2" --clip-grad 1"
OPTS2=$OPTS2" --seed 3407"
OPTS2=$OPTS2" --gradient-accumulation 1"
OPTS2=$OPTS2" --epochs 30"
OPTS2=$OPTS2" --lr 1e-4"
OPTS2=$OPTS2" --weight-decay 0"
OPTS2=$OPTS2" --warmup-iters 1000"
OPTS2=$OPTS2" --early-stop 6"

# generate feature
CUDA_VISIBLE_DEVICES=${GPUS} torchrun ${DISTRIBUTED_ARGS} run_seq2feat.py ${OPTS1}
# train expression model based on feature
CUDA_VISIBLE_DEVICES=${GPUS} torchrun ${DISTRIBUTED_ARGS} run_task.py ${OPTS2}