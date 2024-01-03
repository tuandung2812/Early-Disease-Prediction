CUDA_VISIBLE_DEVICES=3 python train.py --dataset=mimic3 --task=h --epochs=10 --lr=1e-3 --eval_steps=250
CUDA_VISIBLE_DEVICES=3 python train.py --dataset=mimic3 --task=diabetes --epochs=10 --lr=1e-3 --eval_steps=250
# CUDA_VISIBLE_DEVICES=1 python train_simgrace.py