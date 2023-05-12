#!/bin/bash
for u in 0.01 0.1 1
do
  for lr in 0.1 1 10
  do
    python main.py --dataset Cifar10 --optimizer zero_order --u $u --lr $lr | tee -a output.log
    echo "\n" | tee -a output.log
  done
done