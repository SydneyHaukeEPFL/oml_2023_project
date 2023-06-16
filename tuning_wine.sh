#!/bin/bash
for u in 0.000001 0.00001 0.0001
do
  for lr in 0.000001 0.00001 0.0001
  do
    for bs in 4 32 256
    do
      python main.py --dataset wine --model mlp --optimizer zero_order --epochs 500 --u $u --lr $lr --batch_size $bs 
      echo "\n" | tee -a output.log
    done
  done
done
for lr in 0.000001 0.00001 0.0001
do
  for bs in 4 32 256
  do
    python main.py --dataset wine --model mlp --optimizer sgd --epochs 500 --lr $lr --batch_size $bs 
    echo "\n" | tee -a output.log
  done
done