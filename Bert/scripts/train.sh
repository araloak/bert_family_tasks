#!/bin/bash

python ../codes/main.py \
  --times 100 \
  --epoch 2 \
  --batch_size 1 \
  --maxlen 512 \
  --do_train  \
  --do_predict \
  --nclass 3 \
