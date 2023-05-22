#* env for devbox
export NNODES=2
export NGPUS_PER_NODE=8
export JOB_ID=8848
export HOST_NODE_ADDR=10.43.151.23:29400
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens32
export NCCL_IB_DISABLE=1

#* CIFAR10
# torchrun \
# --nnodes=$NNODES --nproc-per-node=$NGPUS_PER_NODE \
# --rdzv_id=$JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR \
# main.py \
# --config=cifar10.yml \
# --exp=cifar10 \
# --doc=cifar10 \
# --ni

#* churches
torchrun \
--nnodes=$NNODES --nproc-per-node=$NGPUS_PER_NODE \
--rdzv_id=$JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR \
main.py \
--config=zhiwen_lsun_256x256_raw.yml \
--exp=zhiwen_lsun_256x256_raw \
--doc=zhiwen_lsun_256x256_raw \
--ni