#!/bin/bash

python3 preprocessing.py \
   --path data/tom_easy/no_noise \
   --all data/tom_easy \
   --c_max_len 8 \
   --output_path preprocessed/tom_easy_no_noise

python3 preprocessing.py \
   --path data/tom_easy/noise \
   --all data/tom_easy \
   --c_max_len 13 \
   --output_path preprocessed/tom_easy_noise

python3 preprocessing.py \
   --path data/tom_easy/noise_at_test \
   --all data/tom_easy \
   --c_max_len 13 \
   --output_path preprocessed/tom_easy_noise_at_test

python3 preprocessing.py \
  --path data/tom/no_noise \
  --all data/tom \
  --c_max_len 40 \
  --output_path preprocessed/tom_no_noise

python3 preprocessing.py \
  --path data/tom/noise \
  --all data/tom \
  --c_max_len 47 \
  --output_path preprocessed/tom_noise

python3 preprocessing.py \
   --path data/tom/noise_at_test \
   --all data/tom \
   --c_max_len 40 \
   --output_path preprocessed/tom_noise_at_test
