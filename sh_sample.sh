torchrun --nnodes=1 --nproc-per-node=8 main.py \
--config=cifar10.yml \
--exp=cifar10 \
--doc=cifar10 \
--ni --sample --fid --timesteps=50 --eta=0