#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu 
#SBATCH --gres=gpu:1

python ../codes/train_with_bert.py \
<<<<<<< HEAD
  --times 100 \
  --epoch 2 \
=======
  --times 10 \
  --epoch 1 \
>>>>>>> 优化了代码
  --batch_size 1 \
  --maxlen 512 \
  --do_train  \
  --do_predict \
  --nclass 3 \
