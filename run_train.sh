#!/bin/bash


NV_GPU=0 nvidia-docker run -t \
  --user $(id -u) \
  -v $(pwd):/src \
  tensorflow/tensorflow:1.4.0-gpu-py3 \
  sh -c \
  "export PYTHONPATH=/src/ && python ../src/train.py \
  --save_path ../src/results  \
  --train_data_path ../src/preprocessed/tom_easy_noise_13_8_11_130_75_31  \
  --test_data_path ../src/preprocessed/tom_easy_noise_13_8_11_130_75_31  \
  --batch_size 32 \
  --q_hidden_size 16 \
  --s_hidden_size 16 \
  --learning_rate 0.002 \
  --max_train_iters 1500 \
  --iter_time 20 \
  --display_step 100"


NV_GPU=1 nvidia-docker run -t \
  --user $(id -u) \
  -v $(pwd):/src \
  tensorflow/tensorflow:1.4.0-gpu-py3 \
  sh -c \
  "export PYTHONPATH=/src/ && python ../src/train.py \
  --save_path ../src/results  \
  --train_data_path ../src/preprocessed/tom_noise_47_8_11_130_75_31 \
  --test_data_path ../src/preprocessed/tom_noise_47_8_11_130_75_31 \
  --batch_size 32 \
  --q_hidden_size 16 \
  --s_hidden_size 16 \
  --learning_rate 0.002 \
  --max_train_iters 2500 \
  --iter_time 20 \
  --display_step 100"


NV_GPU=2 nvidia-docker run -t \
  --user $(id -u) \
  -v $(pwd):/src \
  tensorflow/tensorflow:1.4.0-gpu-py3 \
  sh -c \
  "export PYTHONPATH=/src/ && python ../src/train.py \
  --save_path ../src/results  \
  --train_data_path ../src/preprocessed/tom_easy_noise_at_test_13_8_11_130_75_31 \
  --test_data_path ../src/preprocessed/tom_easy_noise_at_test_13_8_11_130_75_31 \
  --batch_size 32 \
  --q_hidden_size 16 \
  --s_hidden_size 16 \
  --learning_rate 0.002 \
  --max_train_iters 1500 \
  --iter_time 20 \
  --display_step 100"


NV_GPU=2 nvidia-docker run -t \
  --user $(id -u) \
  -v $(pwd):/src \
  tensorflow/tensorflow:1.4.0-gpu-py3 \
  sh -c \
  "export PYTHONPATH=/src/ && python ../src/train.py \
  --save_path ../src/results  \
  --train_data_path ../src/preprocessed/tom_noise_at_test_40_8_11_130_75_31 \
  --test_data_path ../src/preprocessed/tom_noise_at_test_40_8_11_130_75_31 \
  --batch_size 32 \
  --q_hidden_size 16 \
  --s_hidden_size 16 \
  --learning_rate 0.002 \
  --max_train_iters 2500 \
  --iter_time 20 \
  --display_step 100"


NV_GPU=3 nvidia-docker run -t \
  --user $(id -u) \
  -v $(pwd):/src \
  tensorflow/tensorflow:1.4.0-gpu-py3 \
  sh -c \
  "export PYTHONPATH=/src/ && python ../src/train.py \
  --save_path ../src/results  \
  --train_data_path ../src/preprocessed/tom_easy_no_noise_8_8_11_130_75_31 \
  --test_data_path ../src/preprocessed/tom_easy_no_noise_8_8_11_130_75_31 \
  --batch_size 32 \
  --q_hidden_size 16 \
  --s_hidden_size 16 \
  --learning_rate 0.002 \
  --max_train_iters 1500 \
  --iter_time 20 \
  --display_step 100"


NV_GPU=3 nvidia-docker run -t \
  --user $(id -u) \
  -v $(pwd):/src \
  tensorflow/tensorflow:1.4.0-gpu-py3 \
  sh -c \
  "export PYTHONPATH=/src/ && python ../src/train.py \
  --save_path ../src/results  \
  --train_data_path ../src/preprocessed/tom_no_noise_40_8_11_130_75_31 \
  --test_data_path ../src/preprocessed/tom_no_noise_40_8_11_130_75_31 \
  --batch_size 32 \
  --q_hidden_size 16 \
  --s_hidden_size 16 \
  --learning_rate 0.002 \
  --max_train_iters 2500 \
  --iter_time 20 \
  --display_step 100"
