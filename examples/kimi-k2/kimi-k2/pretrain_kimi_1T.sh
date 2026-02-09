#!/bin/bash
set -eo pipefail
set -x
# Runs the "175B" parameter model
# source /mnt/moer-train/public/1000B/TransformerEngine/install_wheel.sh
# Please change the following envrioment variables
# base on the cluster configuration

WORK_HOME=/mnt/moer-train/public/suzy/dev-0205/MT-MegatronLM/examples/kimi-k2
PATCH_HOME=/mnt/moer-train/public/suzy/dev-0205/MT-MegatronLM

TP_SIZE=${TP:-1}
PP_SIZE=${PP:-31}
EP_SIZE=${EP:-8}
FP8=${FP8:-false}
MTP=${MTP:-0}
FORCE_LB=${FORCE_LB:-false}
EXIT_INTERVAL=${EXIT_INTERVAL:-20000000000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}

NUM_NODES=${WORLD_SIZE:-1}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

TOKENIZED_MODEL=/mnt/moer-train/public/models/zjllm-llama3-tokenizer
CURRENT_TIME=$(date "+%Y-%m-%d_%H%M")
RDZV_ID=$CURRENT_TIME
EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}_mtp${MTP}_forcelb${FORCE_LB}_pertensor${FP8}_NO_LOSS_REDUCE"

export ENABLE_PROFILER=0
export PROFILER_FREQ=250
export PROFILER_SAVE_DIR=/mnt/moer-train/public/suzy/output/trace/noforce/
export PROFILER_WITH_STACK=1
# export MUSA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_EXECUTION_TIMEOUT=480000
# export MUSA_EXECUTION_TIMEOUT=86400000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MCCL_IB_GID_INDEX=3
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_TC=136
export MCCL_IB_QPS_PER_CONNECTION=16
export MCCL_CROSS_NIC=0
export USE_RECOMPUTE_VARIANCE=0
export ENABLE_D2H_IN_PERMUTATION=0
export NO_LOSS_REDUCE=1
# export USE_MUSA_MOE=1
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
export MCCL_LIB=/usr/local/musa/lib/libmccl.so
export MUSA_ERROR_DUMP_PATH=/mnt/moer-train/public/suzy/output/logs021
MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH
export USE_DEEPEP_ACE=1
export EP_BALANCE_INFO=0
export MUSA_LOG=0x1

if [ ! -d "${MEGATRON_PATH}/build" ]; then
    cd "${MEGATRON_PATH}"
    python setup.py build_ext --inplace
    cd -
fi

mkdir -p $CHECKPOINT_PATH
# DATA_PATH=$DATA_DIR

#VMM
export PYTORCH_MUSA_ALLOC_CONF="expandable_segments:True"
export TORCH_MCCL_AVOID_RECORD_STREAMS=1

export NODE_ADDR=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)
export GPUS_PER_NODE=8
export NODE_RANK=$RANK
# export MUSA_LAUNCH_BLOCKING=1
# export MCCL_DEBUG=INFO
DATASET_FILE=$DATASET_FILE
DATA_PATH="$(grep -v '^#' ${DATASET_FILE})"
DATA_CACHE_PATH=$DATA_CACHE_PATH
# TOTAL_TOKENS=20934786631107 # 

cd "$(dirname "$0")"
TOTAL_TOKENS="$(($(python3 sum_row.py ${DATASET_FILE})))"

# TOTAL_TOKENS=20932294838446

SEQ_LEN=4096
SAMPLE_SIZE="$((${TOTAL_TOKENS}/${SEQ_LEN}))"
TRAIN_SAMPLES=$SAMPLE_SIZE
TRAIN_ITERS=$(( ${TOTAL_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
WARMUP_STEPS=500
WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))
WSD_DECAY_SAMPLES=$((TRAIN_SAMPLES * 0 / 100)) # 0.35

RUN_DIR=${OUTPUT_DIR}/${CURRENT_TIME}/
mkdir -p "${RUN_DIR}"
LOG_FILE_TMP="${RUN_DIR}/${EXPNAME}.RANK${NODE_RANK}.${NODE_ADDR}.log"
LOG_FILE=${LOG_FILE:-${LOG_FILE_TMP}}

LOG_PATH=$OUTPUT_DIR/${CURRENT_TIME}/logs/$RDZV_ID
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$OUTPUT_DIR/tf_logs/${CURRENT_TIME}/
mkdir -p $TB_PATH
WB_PATH=$OUTPUT_DIR/wandb/${CURRENT_TIME}/
mkdir -p $WB_PATH

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
#     --log_dir $LOG_FILE_TMP #$WORK_HOME/output_log/$RDZV_ID/$EXPNAME/$NODE_RANK
#     --redirects 3
)
    
LAYERS=${LAYERS:-63}
MODEL_ARGS=(
    --num-layers $LAYERS  # 61 
    --hidden-size 7168
    --num-attention-heads 64
    --seq-length 4096 
    --max-position-embeddings 4096 
    --norm-epsilon 1e-6 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --disable-bias-linear 
    --vocab-size 163840 #163840
    --ffn-hidden-size 18432
    --position-embedding-type rope
    --swiglu 
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights
    --rope-type yarn
)

# 24414062 1T
TRAINING_ARGS=(
    --seed 42 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE  
    --train-samples $TRAIN_SAMPLES #2441406200 
    --init-method-std  0.006 
    --use-mcore-models 
    # --no-gradient-accumulation-fusion 
    --no-bias-dropout-fusion
    # --no-rope-fusion
    # --no-bias-swiglu-fusion
    --use-distributed-optimizer 
    --use-flash-attn 
    --sequence-parallel 
    # =================== recompute full schema =================
    # --recompute-granularity full 
    # --recompute-method uniform
    # --recompute-num-layers ${MP_AC_LAYERS:-1}
    # ===========================================================
    --distributed-backend nccl
    --multi-latent-attention
    --qk-layernorm
    --enable-experimental
    # --mlp-recompute
    # --mlp-rms-recompute
    # --recompute-variance
    # --attn-recompute
    # --mla-rms-recompute
    # =================== recompute selective schema ====================
    --recompute-granularity selective
    --recompute-modules mla_up_proj moe_act layernorm mlp
    --offload-moe-fc1-input
    --offload-moe-fused-swiglu-input
    # ==================================================================
    --manual-gc
    --manual-gc-interval 30
    --exit-interval ${EXIT_INTERVAL}
)

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 1
    --mscale 1.0
    --mscale-all-dim 1.0
    # --rotary-base 50000
    # --beta-fast 1.0
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
)

LEARNING_RATE_ARGS=(
    --lr 2e-4  
    --lr-warmup-samples ${WARMUP_SAMPLES}
    --lr-decay-style WSD
    --lr-wsd-decay-style cosine
    --lr-wsd-decay-samples ${WSD_DECAY_SAMPLES}
    --min-lr 2e-5
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP_SIZE  
	--pipeline-model-parallel-size $PP_SIZE 
    # --num-virtual-stages-per-pipeline-rank 2
)

if [[ ${PP_LAYOUT:-none} != none ]]; then
    MODEL_PARALLEL_ARGS+=(
        --pipeline-model-parallel-layout ${PP_LAYOUT}
    )
else
    MODEL_PARALLEL_ARGS+=(
        --decoder-last-pipeline-num-layers 1
    )    
fi

MIXED_PRECISION_ARGS=(
    --bf16 
    --attention-softmax-in-fp32 
    --no-masked-softmax-fusion 
    --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type NullTokenizer
    --tokenizer-model ${TOKENIZED_MODEL}
    --data-cache-path $DATA_CACHE_PATH
    --split 100,0,0
    # --distributed-timeout-minutes 1200
    --distributed-timeout-minutes 10
    --num-dataset-builder-threads 16
    --num-workers 2
    # --no-mmap-bin-files
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 10000
    --eval-interval 1 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH
    --use-checkpoint-opt_param-scheduler
    --auto-detect-ckpt-format
    --eval-iters 0
    --tensorboard-dir $TB_PATH 
)

AUX_LOSS_COEF=${AUX_LOSS_COEF:-0.001}
ROUTER_WARM_STEPS=${ROUTER_WARM_STEPS:-0}

NUM_EXPERTS=${NUM_EXPERTS:-384}
EXPERT_SIZE=${EXPERT_SIZE:-2560}
FIRST_K_DENSE=${FIRST_K_DENSE:-1}
MOE_ROUTER_LOAD_BALANCING_TYPE=${MOE_ROUTER_LOAD_BALANCING_TYPE:-seq_aux_loss}
MOE_ROUTER_SCORE_FUNCTION=${MOE_ROUTER_SCORE_FUNCTION:-sigmoid}

NUM_LAYERS=$(echo "${MODEL_ARGS[@]}" | grep -oP '(?<=--num-layers )\d+')
NUM_LAYERS_MINUS_ONE=$((NUM_LAYERS - $FIRST_K_DENSE))
MOE_LAYER_FREQ="([0]*$FIRST_K_DENSE+[1]*${NUM_LAYERS_MINUS_ONE})*1"
MOE_ARGS=(
    --num-experts $NUM_EXPERTS # 384
    --expert-model-parallel-size $EP_SIZE
    # --moe-router-num-groups 8
    # --moe-router-group-topk 4
    --moe-router-topk 8
    --moe-router-score-function $MOE_ROUTER_SCORE_FUNCTION
    --moe-router-pre-softmax
    --moe-z-loss-coeff 0.0001
    --moe-router-topk-scaling-factor 2.827
    --moe-ffn-hidden-size $EXPERT_SIZE #2048
    --moe-shared-expert-intermediate-size $EXPERT_SIZE #2048
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-grouped-gemm
    # --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    # --moe-router-force-load-balancing
    # --overlap-moe-expert-parallel-comm
    # --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    # --moe-enable-deepep
    # --moe-token-dispatcher-type flex
    # --rotary_seq_len_interpolation_factor 1
    # --router_logit_var_mointor_freq 10
    # --router_maxvio_mointor_freq 10
    # --router_prob_var_mointor_freq 10

    --moe-router-load-balancing-type $MOE_ROUTER_LOAD_BALANCING_TYPE
    --moe-aux-loss-coeff $AUX_LOSS_COEF
)

TOKEN_DISPATCHER=${TOKEN_DISPATCHER:-alltoall}
DEEP_NUM_SM=${DEEP_NUM_SM:-20}
if [ $TOKEN_DISPATCHER = alltoall ]; then
    MOE_ARGS+=(
        --moe-token-dispatcher-type alltoall
    )
elif [ $TOKEN_DISPATCHER = deepep ]; then
    MOE_ARGS+=(
        --moe-token-dispatcher-type flex
        --moe-enable-deepep
        # --moe-deepep-num-sms $DEEP_NUM_SM
    )
fi

if [ $FORCE_LB = true ]; then
    MOE_ARGS+=(--moe-router-force-load-balancing)
fi



TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    # --fp8-format e4m3
    # --fp8-param-gather
    # --fp8-recipe mxfp8
)

if [ $FP8 = true ]; then
    TRANSFORMER_ENGINE_ARGS+=(
        --fp8-format e4m3
        --fp8-param-gather
        --fp8-recipe mxfp8
    )
fi

OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-false}
if [ $OFFLOAD_OPTIMIZER = true ]; then
    NEW_ARGS+=(
        --use-precision-aware-optimizer
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
        --use-torch-optimizer-for-cpu-offload
    )
fi

VPP=${VPP:-off}
if [ $VPP = on ]; then
    NEW_ARGS+=(
        --num-virtual-stages-per-pipeline-rank 2
    )
fi

MULTI_TOKEN_PREDICTION_ARGS=(
    # --use-multi-token-prediction
    # --mtp-coeff 1e-4
    --mtp-num-layers ${MTP}
    --mtp-loss-scaling-factor 0.1
)

if [ ${CPT_TASK:-false} = true ]; then
    EVAL_AND_LOGGING_ARGS+=(
        --no-load-optim
        --no-load-rng
    )
fi

unset MLFLOW_TRACKING_URI
unset MCCL_IB_HCA

torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_kimi.py \
        ${MODEL_ARGS[@]} \
        ${MULTI_TOKEN_PREDICTION_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]} \
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${MLA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${NEW_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]} 2>&1 | tee ${LOG_FILE}
set +x
