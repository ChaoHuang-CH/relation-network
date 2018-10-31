#!/bin/bash

DATASET_REPO_DIR="${HOME}/repos/tom-qa-dataset"

mkdir -p data/
mkdir -p data/tom
mkdir -p data/tom_easy

cp -r ${DATASET_REPO_DIR}/data/tom/world_large_nex_1000_0 data/tom/no_noise
cp -r ${DATASET_REPO_DIR}/data/tom/world_large_nex_1000_10 data/tom/noise_at_test
cp -r ${DATASET_REPO_DIR}/data/tom_train_noise/world_large_nex_1000_10 data/tom/noise

cp -r ${DATASET_REPO_DIR}/data/tom_easy/world_large_nex_1000_0 data/tom_easy/no_noise
cp -r ${DATASET_REPO_DIR}/data/tom_easy/world_large_nex_1000_10 data/tom_easy/noise_at_test
cp -r ${DATASET_REPO_DIR}/data/tom_easy_train_noise/world_large_nex_1000_10 data/tom_easy/noise
